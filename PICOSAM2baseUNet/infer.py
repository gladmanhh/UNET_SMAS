import argparse
import shutil
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from common import Size2D, discover_frames, normalize_output_name, resolve_frames_dir, tensor_from_image
from model import PicoSAM2BaseUNet, reparameterize_model

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = BASE_DIR / "checkpoints" / "picosam2_unet_320x192.pt"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "outputs"
CLASS_NAMES = ["background", "dermis", "smas", "bone"]
DISPLAY_LABELS = {
    1: "dermis",
    2: "subc",
    3: "smas",
    4: "muscle",
    5: "bone",
}
DISPLAY_COLORS = {
    1: (0, 220, 255, 58),
    2: (90, 235, 120, 50),
    3: (255, 220, 0, 72),
    4: (255, 145, 60, 50),
    5: (255, 64, 160, 62),
}
DEFAULT_SMAS_MAX_SLOPE_DEG = 15.0
DEFAULT_SMAS_MAX_ABS_SLOPE = float(np.tan(np.deg2rad(DEFAULT_SMAS_MAX_SLOPE_DEG)))
DEFAULT_SMAS_MIN_MAIN_COMPONENT_RATIO = 0.015

RENDER_ALPHA_LUT = np.zeros(256, dtype=np.uint8)
RENDER_BGR_LUT = np.zeros((256, 3), dtype=np.uint8)
for _class_id, _rgba in DISPLAY_COLORS.items():
    RENDER_ALPHA_LUT[_class_id] = _rgba[3]
    RENDER_BGR_LUT[_class_id] = (_rgba[2], _rgba[1], _rgba[0])
RENDER_INVERSE_ALPHA_LUT = 255 - RENDER_ALPHA_LUT
RENDER_PREMULTIPLIED_BGR_LUT = (
    (
        RENDER_BGR_LUT.astype(np.uint16)
        * RENDER_ALPHA_LUT[:, None].astype(np.uint16)
        + 127
    )
    // 255
).astype(np.uint8).T.copy()


def cap_smas_abs_slope(max_abs_slope: float) -> float:
    value = abs(float(max_abs_slope))
    if value <= 0:
        return DEFAULT_SMAS_MAX_ABS_SLOPE
    return min(value, DEFAULT_SMAS_MAX_ABS_SLOPE)


def load_model(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    class_names = ckpt_args.get("class_names", CLASS_NAMES)
    state = ckpt["model"]
    # A checkpoint saved after reparameterization already holds fused conv weights.
    already_deploy = any("reparam_conv" in key for key in state)
    model = PicoSAM2BaseUNet(
        base_channels=int(ckpt_args.get("base_channels", 32)),
        out_channels=int(ckpt_args.get("num_classes", len(class_names))),
        inference_mode=already_deploy,
        num_conv_branches=int(ckpt_args.get("num_conv_branches", 1)),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    if not already_deploy:
        # Collapse the multi-branch training graph into single convs for speed.
        model = reparameterize_model(model, inplace=True).to(device)
        model.eval()
    return model, ckpt_args


@torch.no_grad()
def predict_prepared_class_map(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    tensor = tensor[None, :, :, :].to(device)
    logits = model(tensor)
    return torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)


def predict_class_map(model: torch.nn.Module, image: Image.Image, image_size: Size2D, device: torch.device) -> np.ndarray:
    return predict_prepared_class_map(model, tensor_from_image(image, image_size), device)


def load_and_prepare_frame(frame_path: Path, image_size: Size2D) -> tuple[Image.Image, torch.Tensor, float]:
    start = time.perf_counter()
    with Image.open(frame_path) as opened:
        image = opened.convert("RGB")
    tensor = tensor_from_image(image, image_size)
    return image, tensor, time.perf_counter() - start


def scale_length(value: float, scale: float, minimum: int = 1) -> int:
    return max(int(minimum), int(round(float(value) * float(scale))))


def scale_odd_window(value: int, scale: float, minimum: int = 3) -> int:
    if int(value) <= 1:
        return 1
    scaled = scale_length(value, scale, minimum)
    return scaled if scaled % 2 == 1 else scaled + 1


def resolve_inference_size(value: str | None, ckpt_args: dict) -> Size2D:
    if value is None:
        return Size2D(int(ckpt_args.get("width", 384)), int(ckpt_args.get("height", 224)))
    parts = value.lower().replace(" ", "").split("x")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Expected inference size like 320x192, got: {value}")
    width, height = (int(part) for part in parts)
    if width <= 0 or height <= 0 or width % 16 != 0 or height % 16 != 0:
        raise ValueError("Inference width and height must be positive multiples of 16.")
    return Size2D(width, height)


def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1:
        return values.astype(np.float32, copy=True)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def median_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1:
        return values.astype(np.float32, copy=True)
    if window % 2 == 0:
        window += 1
    if ndi is not None:
        return ndi.median_filter(values.astype(np.float32), size=window, mode="nearest").astype(np.float32)
    pad = window // 2
    padded = np.pad(values.astype(np.float32), (pad, pad), mode="edge")
    return np.array([np.median(padded[i : i + window]) for i in range(len(values))], dtype=np.float32)


def clamp_line_slope(values: np.ndarray, max_abs_slope: float) -> np.ndarray:
    max_abs_slope = cap_smas_abs_slope(max_abs_slope)
    out = values.astype(np.float32, copy=True)
    for i in range(1, len(out)):
        out[i] = np.clip(out[i], out[i - 1] - max_abs_slope, out[i - 1] + max_abs_slope)
    for i in range(len(out) - 2, -1, -1):
        out[i] = np.clip(out[i], out[i + 1] - max_abs_slope, out[i + 1] + max_abs_slope)
    return out


def estimate_edge_slope(xs: np.ndarray, ys: np.ndarray, max_abs_slope: float) -> float:
    if len(xs) < 2:
        return 0.0
    max_abs_slope = cap_smas_abs_slope(max_abs_slope)
    slope = float(np.polyfit(xs.astype(np.float32), ys.astype(np.float32), 1)[0])
    return float(np.clip(slope, -max_abs_slope, max_abs_slope))


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    mask_u8 = mask.astype(np.uint8)
    if cv2 is not None:
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
        components = []
        for label_id in range(1, count):
            x, y, w, h, area = stats[label_id]
            cx, cy = centroids[label_id]
            components.append(
                {
                    "id": label_id,
                    "area": int(area),
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "cx": float(cx),
                    "cy": float(cy),
                    "touches_bottom": int(y + h) >= mask.shape[0],
                }
            )
        return labels, components

    if ndi is not None:
        labels, count = ndi.label(mask_u8, structure=np.ones((3, 3), dtype=np.uint8))
        objects = ndi.find_objects(labels)
        components = []
        for label_id, slc in enumerate(objects, start=1):
            if slc is None:
                continue
            ys, xs = np.where(labels[slc] == label_id)
            if len(xs) == 0:
                continue
            y0 = int(slc[0].start)
            x0 = int(slc[1].start)
            xs_abs = xs + x0
            ys_abs = ys + y0
            y_min = int(ys_abs.min())
            y_max = int(ys_abs.max())
            x_min = int(xs_abs.min())
            x_max = int(xs_abs.max())
            components.append(
                {
                    "id": label_id,
                    "area": int(len(xs_abs)),
                    "x": x_min,
                    "y": y_min,
                    "w": int(x_max - x_min + 1),
                    "h": int(y_max - y_min + 1),
                    "cx": float(xs_abs.mean()),
                    "cy": float(ys_abs.mean()),
                    "touches_bottom": y_max >= mask.shape[0] - 1,
                }
            )
        return labels, components

    labels = np.zeros(mask.shape, dtype=np.int32)
    return labels, []


def fill_region_below_boundary(mask: np.ndarray, smooth_window: int) -> np.ndarray:
    height, width = mask.shape
    counts = mask.sum(axis=0)
    valid_x = np.where(counts > 0)[0]
    if len(valid_x) < 2:
        return mask.copy()

    top = np.full(width, np.nan, dtype=np.float32)
    top[valid_x] = np.argmax(mask[:, valid_x], axis=0).astype(np.float32)

    x_all = np.arange(width, dtype=np.float32)
    boundary = np.interp(x_all, valid_x.astype(np.float32), top[valid_x])
    boundary = median_1d(boundary, smooth_window)
    boundary = smooth_1d(boundary, max(5, smooth_window // 2))
    boundary = np.clip(boundary, 0, height - 1)

    boundary_y = np.rint(boundary).astype(np.int32)
    return np.arange(height, dtype=np.int32)[:, None] >= boundary_y[None, :]


def postprocess_bone_region(
    class_arr: np.ndarray,
    min_area_ratio: float = 0.002,
    min_centroid_y_ratio: float = 0.55,
    boundary_smooth_window: int = 51,
) -> np.ndarray:
    bone = class_arr == 3
    if not bone.any():
        return class_arr

    height, width = bone.shape
    labels, components = connected_components(bone)
    if not components:
        return class_arr

    min_area = max(1, int(round(height * width * min_area_ratio)))
    min_centroid_y = height * float(min_centroid_y_ratio)
    eligible = [c for c in components if c["area"] >= min_area and c["cy"] >= min_centroid_y]
    if not eligible:
        result = class_arr.copy()
        result[result == 3] = 0
        return result

    bottom_components = [c for c in eligible if c["touches_bottom"]]
    selected = max(bottom_components or eligible, key=lambda c: (c["area"], c["cy"]))
    selected_mask = labels == selected["id"]
    filled_bone = fill_region_below_boundary(selected_mask, boundary_smooth_window)

    result = class_arr.copy()
    result[result == 3] = 0
    protected = (result == 1) | (result == 2)
    result[filled_bone & ~protected] = 3
    return result


def clean_smas_components(
    smas: np.ndarray,
    min_area_ratio: float = 0.001,
    max_y_gap_ratio: float = 0.12,
    min_main_area_ratio: float = DEFAULT_SMAS_MIN_MAIN_COMPONENT_RATIO,
) -> np.ndarray:
    labels, components = connected_components(smas)
    if not components:
        return smas

    height, width = smas.shape
    min_main_area = max(1, int(round(height * width * min_main_area_ratio)))
    min_area = max(1, int(round(height * width * min_area_ratio)))
    max_y_gap = height * float(max_y_gap_ratio)
    anchor = max(components, key=lambda c: (c["area"], c["w"]))
    if anchor["area"] < min_main_area:
        return np.zeros_like(smas, dtype=bool)
    if len(components) <= 1:
        return smas

    keep_ids = []
    for component in components:
        if component["id"] == anchor["id"]:
            keep_ids.append(component["id"])
            continue
        if component["area"] < min_area:
            continue
        if abs(component["cy"] - anchor["cy"]) > max_y_gap:
            continue
        keep_ids.append(component["id"])

    if not keep_ids:
        return smas
    return np.isin(labels, keep_ids)


def despike_layer_center(
    center: np.ndarray,
    spike_window: int,
    max_deviation: float,
    max_abs_slope: float,
    smooth_window: int,
) -> np.ndarray:
    max_abs_slope = cap_smas_abs_slope(max_abs_slope)
    baseline = median_1d(center, spike_window)
    baseline = smooth_1d(baseline, max(5, smooth_window // 2))
    slope = np.gradient(center.astype(np.float32))
    baseline_slope = np.gradient(baseline.astype(np.float32))
    bad = (np.abs(center - baseline) > max_deviation) | (
        np.abs(slope - baseline_slope) > max_abs_slope * 3.0
    )
    if bad.any():
        bad = ndi.binary_dilation(bad, iterations=max(1, spike_window // 16)) if ndi is not None else bad
        center = center.copy()
        center[bad] = baseline[bad]
    center = clamp_line_slope(center, max_abs_slope)
    return smooth_1d(center, max(5, smooth_window // 2))


def connect_smas_to_left(
    class_arr: np.ndarray,
    min_column_pixels: int = 8,
    min_valid_columns: int = 8,
    slope_window: int = 90,
    smooth_window: int = 31,
    max_abs_slope: float = DEFAULT_SMAS_MAX_ABS_SLOPE,
    spike_window: int = 91,
    max_center_deviation: float = 24.0,
    band_max_abs_slope: float = DEFAULT_SMAS_MAX_ABS_SLOPE,
    min_component_area_ratio: float = 0.001,
    max_component_y_gap_ratio: float = 0.12,
    min_main_component_area_ratio: float = DEFAULT_SMAS_MIN_MAIN_COMPONENT_RATIO,
    min_band_thickness: float = 8.0,
) -> np.ndarray:
    smas = class_arr == 2
    smas = clean_smas_components(
        smas,
        min_area_ratio=min_component_area_ratio,
        max_y_gap_ratio=max_component_y_gap_ratio,
        min_main_area_ratio=min_main_component_area_ratio,
    )
    if not smas.any():
        result = class_arr.copy()
        result[result == 2] = 0
        return result
    height, width = smas.shape
    counts = smas.sum(axis=0)
    valid_x = np.where(counts >= min_column_pixels)[0]
    if len(valid_x) < max(2, int(min_valid_columns)):
        result = class_arr.copy()
        result[result == 2] = 0
        return result

    top = np.full(width, np.nan, dtype=np.float32)
    bottom = np.full(width, np.nan, dtype=np.float32)
    top[valid_x] = np.argmax(smas[:, valid_x], axis=0).astype(np.float32)
    bottom[valid_x] = (
        height - 1 - np.argmax(smas[::-1, valid_x], axis=0)
    ).astype(np.float32)

    center_valid = (top[valid_x] + bottom[valid_x]) * 0.5
    thick_valid = np.maximum(bottom[valid_x] - top[valid_x] + 1.0, 2.0)

    first = int(valid_x[0])
    last = int(valid_x[-1])
    x_all = np.arange(width, dtype=np.float32)

    center = np.interp(x_all, valid_x.astype(np.float32), center_valid)
    thick = np.interp(x_all, valid_x.astype(np.float32), thick_valid)
    center = smooth_1d(center, smooth_window)
    thick = smooth_1d(thick, smooth_window)

    left_fit_x = valid_x[valid_x <= min(last, first + slope_window)]
    if len(left_fit_x) >= 2 and first > 0:
        left_center = center[left_fit_x]
        center_slope = estimate_edge_slope(left_fit_x, left_center, max_abs_slope)
        left_thick = float(np.median(thick[left_fit_x]))
        for x in range(first - 1, -1, -1):
            center[x] = center[first] + center_slope * float(x - first)
            thick[x] = left_thick

    right_fit_x = valid_x[valid_x >= max(first, last - slope_window)]
    if len(right_fit_x) >= 2 and last < width - 1:
        right_center = center[right_fit_x]
        center_slope = estimate_edge_slope(right_fit_x, right_center, max_abs_slope)
        right_thick = float(np.median(thick[right_fit_x]))
        for x in range(last + 1, width):
            center[x] = center[last] + center_slope * float(x - last)
            thick[x] = right_thick

    center = despike_layer_center(
        center,
        spike_window=spike_window,
        max_deviation=max_center_deviation,
        max_abs_slope=band_max_abs_slope,
        smooth_window=smooth_window,
    )
    thick = median_1d(thick, spike_window)
    min_thickness = max(2.0, float(min_band_thickness))
    max_thickness = max(min_thickness, float(height) * 0.35)
    thick = np.clip(smooth_1d(thick, max(5, smooth_window // 2)), min_thickness, max_thickness)

    half = thick * 0.5
    y1 = np.clip(np.rint(center - half).astype(np.int32), 0, height - 1)
    y2 = np.clip(np.rint(center + half).astype(np.int32), 0, height - 1)
    rows = np.arange(height, dtype=np.int32)[:, None]
    filled = (rows >= y1[None, :]) & (rows <= y2[None, :])

    result = class_arr.copy()
    result[result == 2] = 0
    protected = (result == 1) | (result == 3)
    result[filled & ~protected] = 2
    return result


def boundary_lines(mask: np.ndarray, min_column_pixels: int = 3, smooth_window: int = 31) -> tuple[np.ndarray | None, np.ndarray | None]:
    height, width = mask.shape
    counts = mask.sum(axis=0)
    valid_x = np.where(counts >= min_column_pixels)[0]
    if len(valid_x) < 2:
        return None, None

    top = np.zeros(width, dtype=np.float32)
    bottom = np.zeros(width, dtype=np.float32)
    top_valid = np.argmax(mask[:, valid_x], axis=0).astype(np.float32)
    bottom_valid = (height - 1 - np.argmax(mask[::-1, valid_x], axis=0)).astype(np.float32)
    x_all = np.arange(width, dtype=np.float32)
    top[:] = np.interp(x_all, valid_x.astype(np.float32), top_valid)
    bottom[:] = np.interp(x_all, valid_x.astype(np.float32), bottom_valid)
    top = np.clip(smooth_1d(median_1d(top, smooth_window), max(5, smooth_window // 2)), 0, height - 1)
    bottom = np.clip(smooth_1d(median_1d(bottom, smooth_window), max(5, smooth_window // 2)), 0, height - 1)
    return top, bottom


def fill_between(display_arr: np.ndarray, upper: np.ndarray, lower: np.ndarray, class_id: int) -> None:
    height, width = display_arr.shape
    y1 = np.clip(np.rint(upper).astype(np.int32), 0, height - 1)
    y2 = np.clip(np.rint(lower).astype(np.int32), 0, height - 1)
    rows = np.arange(height, dtype=np.int32)[:, None]
    fill = (rows >= y1[None, :]) & (rows <= y2[None, :]) & (y2[None, :] >= y1[None, :])
    display_arr[fill & (display_arr == 0)] = class_id


def build_display_layers(
    class_arr: np.ndarray,
    min_column_pixels: int = 3,
    smooth_window: int = 31,
) -> np.ndarray:
    dermis = class_arr == 1
    smas = class_arr == 2
    bone = class_arr == 3
    display_arr = np.zeros(class_arr.shape, dtype=np.uint8)

    dermis_top, dermis_bottom = boundary_lines(dermis, min_column_pixels, smooth_window)
    smas_top, smas_bottom = boundary_lines(smas, min_column_pixels, smooth_window)
    bone_top, bone_bottom = boundary_lines(bone, min_column_pixels, smooth_window)

    if dermis_bottom is not None and smas_top is not None:
        fill_between(display_arr, dermis_bottom + 1, smas_top - 1, 2)
    if smas_bottom is not None and bone_top is not None:
        fill_between(display_arr, smas_bottom + 1, bone_top - 1, 4)

    display_arr[dermis] = 1
    display_arr[smas] = 3
    display_arr[bone] = 5
    return display_arr


def draw_layer_labels_cv2(frame_bgr: np.ndarray, display_arr: np.ndarray) -> None:
    frame_height, frame_width = frame_bgr.shape[:2]
    mask_height, mask_width = display_arr.shape
    scale_x = frame_width / float(mask_width)
    scale_y = frame_height / float(mask_height)
    min_label_pixels = max(1, int(round(400.0 / (scale_x * scale_y))))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness = 1
    pad_x = 5
    pad_y = 3

    for class_id, label in DISPLAY_LABELS.items():
        ys, xs = np.where(display_arr == class_id)
        if len(xs) < min_label_pixels:
            continue
        cx = int(round((float(np.median(xs)) + 0.5) * scale_x))
        cy = int(round((float(np.median(ys)) + 0.5) * scale_y))
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        box_w = text_w + pad_x * 2
        box_h = text_h + baseline + pad_y * 2
        x1 = max(2, min(frame_width - box_w - 2, cx - box_w // 2))
        y1 = max(2, min(frame_height - box_h - 2, cy - box_h // 2))
        x2 = x1 + text_w + pad_x * 2
        y2 = y1 + box_h
        roi = frame_bgr[y1:y2, x1:x2]
        cv2.addWeighted(roi, 1.0 - 88.0 / 255.0, np.zeros_like(roi), 0.0, 0.0, dst=roi)
        text_origin = (x1 + pad_x, y1 + pad_y + text_h)
        cv2.putText(frame_bgr, label, text_origin, font, font_scale, (235, 235, 235), thickness, cv2.LINE_AA)


def overlay_multiclass_cv2(
    image: Image.Image,
    class_arr: np.ndarray,
    display_min_column_pixels: int,
    display_smooth_window: int,
) -> np.ndarray:
    if cv2 is None:
        raise ImportError("OpenCV is required to render the final video.")
    display_low = build_display_layers(class_arr, display_min_column_pixels, display_smooth_window)
    display_full = cv2.resize(
        display_low,
        (image.width, image.height),
        interpolation=cv2.INTER_NEAREST,
    )
    frame_bgr = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    inverse_alpha = cv2.LUT(display_full, RENDER_INVERSE_ALPHA_LUT)
    inverse_alpha_bgr = cv2.merge((inverse_alpha, inverse_alpha, inverse_alpha))
    frame_bgr = cv2.multiply(frame_bgr, inverse_alpha_bgr, scale=1.0 / 255.0)
    premultiplied_color = cv2.merge(
        tuple(cv2.LUT(display_full, RENDER_PREMULTIPLIED_BGR_LUT[channel]) for channel in range(3))
    )
    frame_bgr = cv2.add(frame_bgr, premultiplied_color)
    draw_layer_labels_cv2(frame_bgr, display_low)
    return frame_bgr


def postprocess_render_and_write(
    image: Image.Image,
    class_arr: np.ndarray,
    args: argparse.Namespace,
    scaled: dict[str, float | int],
    writer: "cv2.VideoWriter",
) -> tuple[float, float, float]:
    post_start = time.perf_counter()
    if args.clean_bone:
        class_arr = postprocess_bone_region(
            class_arr,
            min_area_ratio=args.bone_min_area_ratio,
            min_centroid_y_ratio=args.bone_min_centroid_y_ratio,
            boundary_smooth_window=int(scaled["bone_boundary_smooth_window"]),
        )
    if args.connect_smas_left:
        class_arr = connect_smas_to_left(
            class_arr,
            min_column_pixels=int(scaled["smas_min_column_pixels"]),
            min_valid_columns=int(scaled["smas_min_valid_columns"]),
            slope_window=int(scaled["smas_slope_window"]),
            smooth_window=int(scaled["smas_smooth_window"]),
            max_abs_slope=float(scaled["smas_max_abs_slope"]),
            spike_window=int(scaled["smas_spike_window"]),
            max_center_deviation=float(scaled["smas_max_center_deviation"]),
            band_max_abs_slope=float(scaled["smas_band_max_abs_slope"]),
            min_component_area_ratio=args.smas_min_component_area_ratio,
            max_component_y_gap_ratio=args.smas_max_component_y_gap_ratio,
            min_main_component_area_ratio=args.smas_min_main_component_area_ratio,
            min_band_thickness=float(scaled["smas_min_band_thickness"]),
        )
    post_seconds = time.perf_counter() - post_start

    overlay_start = time.perf_counter()
    rendered_bgr = overlay_multiclass_cv2(
        image,
        class_arr,
        display_min_column_pixels=int(scaled["display_min_column_pixels"]),
        display_smooth_window=int(scaled["display_smooth_window"]),
    )
    overlay_seconds = time.perf_counter() - overlay_start

    write_start = time.perf_counter()
    writer.write(rendered_bgr)
    video_write_seconds = time.perf_counter() - write_start
    return post_seconds, overlay_seconds, video_write_seconds


def cleanup_legacy_outputs(target_root: Path) -> None:
    for name in ["frames", "class_masks", "dermis_masks", "smas_masks", "bone_masks"]:
        path = target_root / name
        if path.exists():
            shutil.rmtree(path)
    csv_path = target_root / "predictions.csv"
    if csv_path.exists():
        csv_path.unlink()
    for video_path in target_root.glob("*.mp4"):
        video_path.unlink()
    for video_path in target_root.glob("*.avi"):
        video_path.unlink()


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def create_video_writer(path: Path, size: tuple[int, int], fps: float, codec: str) -> "cv2.VideoWriter":
    if cv2 is None:
        raise ImportError("OpenCV is required to save the final video. Install opencv-python or python3-opencv.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def run_inference(args: argparse.Namespace) -> Path:
    output_name = normalize_output_name(args.output_id)
    frame_dir = resolve_frames_dir(output_name, args.frames_dir)
    frames = discover_frames(frame_dir)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (BASE_DIR / checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_args = load_model(checkpoint, device)
    image_size = resolve_inference_size(args.inference_size, ckpt_args)
    pipeline_depth = int(args.pipeline_depth)
    if pipeline_depth < 1:
        raise ValueError("Pipeline depth must be at least 1.")

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (BASE_DIR / output_root).resolve()
    target_root = output_root / output_name
    cleanup_legacy_outputs(target_root)
    video_path = target_root / f"{output_name}_segmentation.mp4"
    with Image.open(frames[0]) as opened:
        first_image = opened.convert("RGB")
    writer = create_video_writer(video_path, first_image.size, args.video_fps, args.video_codec)

    scale_x = image_size.width / float(first_image.width)
    scale_y = image_size.height / float(first_image.height)
    slope_scale = scale_y / scale_x
    scaled = {
        "smas_min_column_pixels": scale_length(args.smas_min_column_pixels, scale_y),
        "smas_min_valid_columns": scale_length(8, scale_x, minimum=2),
        "smas_slope_window": scale_length(args.smas_slope_window, scale_x, minimum=2),
        "smas_smooth_window": scale_odd_window(args.smas_smooth_window, scale_x),
        "smas_spike_window": scale_odd_window(args.smas_spike_window, scale_x),
        "smas_max_center_deviation": max(1.0, float(args.smas_max_center_deviation) * scale_y),
        "smas_min_band_thickness": max(2.0, 8.0 * scale_y),
        "smas_max_abs_slope": args.smas_max_abs_slope * slope_scale,
        "smas_band_max_abs_slope": args.smas_band_max_abs_slope * slope_scale,
        "bone_boundary_smooth_window": scale_odd_window(args.bone_boundary_smooth_window, scale_x),
        "display_min_column_pixels": scale_length(3, scale_y),
        "display_smooth_window": scale_odd_window(31, scale_x),
    }

    print(f"[Input] {frame_dir}")
    print(f"[Checkpoint] {checkpoint}")
    print(f"[Output] {target_root}")
    print(f"[Video] {video_path}")
    print(f"[Frames] {len(frames)} device={device}")
    print(
        f"[Pipeline] model+postprocess={image_size.width}x{image_size.height}, "
        f"render={first_image.width}x{first_image.height}, renderer=opencv, depth={pipeline_depth}"
    )

    total_start = time.perf_counter()
    load_preprocess_seconds = 0.0
    model_seconds = 0.0
    postprocess_seconds = 0.0
    overlay_seconds = 0.0
    video_write_seconds = 0.0
    try:
        with (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="frame-loader") as load_pool,
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="frame-output") as output_pool,
            tqdm(total=len(frames), desc=f"segment-{output_name}", unit="frame") as progress,
        ):
            pending_loads = deque()
            pending_outputs = deque()
            next_frame_index = 0

            while next_frame_index < min(pipeline_depth, len(frames)):
                pending_loads.append(load_pool.submit(load_and_prepare_frame, frames[next_frame_index], image_size))
                next_frame_index += 1

            while pending_loads:
                image, tensor, load_seconds = pending_loads.popleft().result()
                load_preprocess_seconds += load_seconds
                if next_frame_index < len(frames):
                    pending_loads.append(load_pool.submit(load_and_prepare_frame, frames[next_frame_index], image_size))
                    next_frame_index += 1

                sync_device(device)
                model_start = time.perf_counter()
                class_arr = predict_prepared_class_map(model, tensor, device)
                sync_device(device)
                model_seconds += time.perf_counter() - model_start

                pending_outputs.append(
                    output_pool.submit(
                        postprocess_render_and_write,
                        image,
                        class_arr,
                        args,
                        scaled,
                        writer,
                    )
                )
                if len(pending_outputs) >= pipeline_depth:
                    post_seconds, render_seconds, write_seconds = pending_outputs.popleft().result()
                    postprocess_seconds += post_seconds
                    overlay_seconds += render_seconds
                    video_write_seconds += write_seconds
                    progress.update(1)

            while pending_outputs:
                post_seconds, render_seconds, write_seconds = pending_outputs.popleft().result()
                postprocess_seconds += post_seconds
                overlay_seconds += render_seconds
                video_write_seconds += write_seconds
                progress.update(1)
    finally:
        writer.release()

    total_seconds = time.perf_counter() - total_start
    frame_count = len(frames)
    print(
        "[Speed] "
        f"total={total_seconds:.3f}s ({frame_count / total_seconds:.2f} fps), "
        f"load_preprocess={load_preprocess_seconds:.3f}s ({frame_count / load_preprocess_seconds:.2f} fps), "
        f"model={model_seconds:.3f}s ({frame_count / model_seconds:.2f} fps), "
        f"postprocess={postprocess_seconds:.3f}s ({frame_count / postprocess_seconds:.2f} fps), "
        f"overlay={overlay_seconds:.3f}s ({frame_count / overlay_seconds:.2f} fps), "
        f"video_write={video_write_seconds:.3f}s ({frame_count / video_write_seconds:.2f} fps)"
    )
    print(f"[Done] {video_path}")
    return target_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer mutually exclusive dermis + SMAS + bone masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("output_id", nargs="?", default=None)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--frames-dir", default=None)
    parser.add_argument(
        "--inference-size",
        default=None,
        metavar="WIDTHxHEIGHT",
        help="Override checkpoint inference size; both dimensions must be multiples of 16.",
    )
    parser.add_argument(
        "--pipeline-depth",
        type=int,
        default=3,
        help="Maximum prepared/output frames kept in the overlapped pipeline.",
    )
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--video-codec", default="mp4v")
    parser.add_argument(
        "--no-connect-smas-left",
        "--no-connect-smas-edges",
        dest="connect_smas_left",
        action="store_false",
        help="Disable SMAS edge connection postprocess.",
    )
    parser.add_argument("--smas-min-column-pixels", type=int, default=8)
    parser.add_argument("--smas-slope-window", type=int, default=90)
    parser.add_argument("--smas-smooth-window", type=int, default=31)
    parser.add_argument(
        "--smas-max-abs-slope",
        type=float,
        default=DEFAULT_SMAS_MAX_ABS_SLOPE,
        help=f"Maximum absolute SMAS extension slope, tan({DEFAULT_SMAS_MAX_SLOPE_DEG:g} deg).",
    )
    parser.add_argument("--smas-spike-window", type=int, default=91)
    parser.add_argument("--smas-max-center-deviation", type=float, default=24.0)
    parser.add_argument(
        "--smas-band-max-abs-slope",
        type=float,
        default=DEFAULT_SMAS_MAX_ABS_SLOPE,
        help=f"Maximum absolute SMAS band slope after smoothing, tan({DEFAULT_SMAS_MAX_SLOPE_DEG:g} deg).",
    )
    parser.add_argument("--smas-min-component-area-ratio", type=float, default=0.001)
    parser.add_argument("--smas-max-component-y-gap-ratio", type=float, default=0.12)
    parser.add_argument(
        "--smas-min-main-component-area-ratio",
        type=float,
        default=DEFAULT_SMAS_MIN_MAIN_COMPONENT_RATIO,
        help="Treat SMAS as absent when the largest SMAS component is below this image-area ratio.",
    )
    parser.add_argument("--no-clean-bone", dest="clean_bone", action="store_false")
    parser.add_argument("--bone-min-area-ratio", type=float, default=0.002)
    parser.add_argument("--bone-min-centroid-y-ratio", type=float, default=0.55)
    parser.add_argument("--bone-boundary-smooth-window", type=int, default=51)
    parser.set_defaults(connect_smas_left=True, clean_bone=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.output_id is None:
        args.output_id = input("Output number (1 -> output1) [25]: ").strip() or "25"
    run_inference(args)


if __name__ == "__main__":
    main()
