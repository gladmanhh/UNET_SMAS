import argparse
import csv
import math
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = SCRIPT_DIR / "outputs"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs_smas"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def normalize_output_name(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("Output id is empty.")
    if text.lower().startswith("output"):
        suffix = text[6:]
        return f"output{int(suffix)}" if suffix.isdigit() else text
    if text.isdigit():
        return f"output{int(text)}"
    raise ValueError(f"Expected a number like 25 or an output name like output25, got: {value}")


def parse_bbox(text: str) -> list[float] | None:
    parts = text.split()
    if len(parts) != 4:
        return None
    try:
        return [float(part) for part in parts]
    except ValueError:
        return None


def clamp_bbox(box: list[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    left = max(0, min(width - 1, int(math.floor(x1))))
    top = max(0, min(height - 1, int(math.floor(y1))))
    right = max(0, min(width, int(math.ceil(x2))))
    bottom = max(0, min(height, int(math.ceil(y2))))
    if right <= left + 2 or bottom <= top + 2:
        return None
    return left, top, right, bottom


def normalize01(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float32, copy=False)
    lo, hi = np.percentile(arr, [3, 97])
    if hi <= lo + 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def make_layer_maps(crop_gray: np.ndarray, band_height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gray_u8 = crop_gray.astype(np.uint8, copy=False)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_u8)
    img = clahe.astype(np.float32) / 255.0

    height, width = img.shape
    bg_sigma_x = max(12.0, width / 40.0)
    bg_sigma_y = max(3.0, height / 6.0)
    local_bg = cv2.GaussianBlur(img, (0, 0), sigmaX=bg_sigma_x, sigmaY=bg_sigma_y)
    bright = normalize01(np.maximum(img - local_bg, 0.0))
    bright = cv2.GaussianBlur(bright, (0, 0), sigmaX=4.0, sigmaY=0.8)

    smooth = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5, sigmaY=0.8)
    d2y = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=5)
    ridge = normalize01(np.maximum(-d2y, 0.0))

    band_height = max(3, min(height, int(band_height)))
    band_kernel = np.ones((band_height, 1), dtype=np.float32) / float(band_height)
    band_bright = cv2.filter2D(bright, cv2.CV_32F, band_kernel, borderType=cv2.BORDER_REPLICATE)
    band_score = normalize01(0.75 * band_bright + 0.25 * ridge)
    band_score = cv2.GaussianBlur(band_score, (0, 0), sigmaX=8.0, sigmaY=1.0)

    edge_base = cv2.GaussianBlur(img, (0, 0), sigmaX=5.0, sigmaY=1.0)
    grad_y = cv2.Sobel(edge_base, cv2.CV_32F, 0, 1, ksize=3)
    top_edge = normalize01(np.maximum(grad_y, 0.0))
    bottom_edge = normalize01(np.maximum(-grad_y, 0.0))
    return normalize01(band_score), normalize01(top_edge), normalize01(bottom_edge), normalize01(bright)


def resize_width(values: np.ndarray, x_stride: int) -> np.ndarray:
    height, width = values.shape
    x_stride = max(1, int(x_stride))
    if x_stride == 1:
        return values
    small_width = max(2, int(math.ceil(width / x_stride)))
    return cv2.resize(values, (small_width, height), interpolation=cv2.INTER_AREA)


def trace_path_fast(
    score: np.ndarray,
    max_step: int,
    smoothness: float,
) -> tuple[np.ndarray, float]:
    height, width = score.shape
    max_step = max(1, int(max_step))
    cost = -score

    dp = np.empty((height, width), dtype=np.float32)
    parent = np.zeros((width, height), dtype=np.int16)
    dp[:, 0] = cost[:, 0]

    for x in range(1, width):
        prev = dp[:, x - 1]
        best = np.full(height, np.inf, dtype=np.float32)
        best_parent = np.zeros(height, dtype=np.int16)
        for dy in range(-max_step, max_step + 1):
            penalty = smoothness * abs(dy)
            candidate = np.full(height, np.inf, dtype=np.float32)
            candidate_parent = np.zeros(height, dtype=np.int16)
            if dy > 0:
                candidate[dy:] = prev[:-dy] + penalty
                candidate_parent[dy:] = np.arange(0, height - dy, dtype=np.int16)
            elif dy < 0:
                candidate[:dy] = prev[-dy:] + penalty
                candidate_parent[:dy] = np.arange(-dy, height, dtype=np.int16)
            else:
                candidate[:] = prev + penalty
                candidate_parent[:] = np.arange(height, dtype=np.int16)

            keep = candidate < best
            best[keep] = candidate[keep]
            best_parent[keep] = candidate_parent[keep]

        dp[:, x] = cost[:, x] + best
        parent[x] = best_parent

    path = np.empty(width, dtype=np.int32)
    path[-1] = int(np.argmin(dp[:, -1]))
    for x in range(width - 1, 0, -1):
        path[x - 1] = int(parent[x, path[x]])

    path_score = float(np.mean(score[path, np.arange(width)]))
    return path, path_score


def smooth_path(path: np.ndarray, sigma: float, height: int) -> np.ndarray:
    if sigma <= 0:
        return np.clip(path, 0, height - 1).astype(np.int32)
    path_f = path.astype(np.float32)[None, :]
    smoothed = cv2.GaussianBlur(path_f, (0, 0), sigmaX=sigma, sigmaY=0)[0]
    return np.clip(np.rint(smoothed), 0, height - 1).astype(np.int32)


def boundary_score(
    edge: np.ndarray,
    bright: np.ndarray,
    center: np.ndarray,
    side: str,
    args: argparse.Namespace,
) -> np.ndarray:
    height, _ = edge.shape
    ys = np.arange(height, dtype=np.float32)[:, None]
    centers = center.astype(np.float32)[None, :]
    half_band = float(args.band_height) / 2.0
    min_gap = max(2.0, float(args.min_thickness) / 2.0)
    prior_sigma = max(2.0, float(args.band_height) / 2.0)

    if side == "top":
        target = centers - half_band
        allowed = (ys >= centers - float(args.max_thickness)) & (ys <= centers - min_gap)
    else:
        target = centers + half_band
        allowed = (ys >= centers + min_gap) & (ys <= centers + float(args.max_thickness))

    prior = np.exp(-0.5 * ((ys - target) / prior_sigma) ** 2).astype(np.float32)
    combined = normalize01(0.72 * edge + 0.28 * bright)
    return np.where(allowed, combined * (0.35 + 0.65 * prior), -1.0).astype(np.float32)


def interpolate_path(path: np.ndarray, width: int, height: int, sigma: float) -> np.ndarray:
    if len(path) == width:
        full = path.astype(np.float32)
    else:
        src_x = np.arange(len(path), dtype=np.float32)
        dst_x = np.linspace(0, len(path) - 1, width, dtype=np.float32)
        full = np.interp(dst_x, src_x, path).astype(np.float32)
    return smooth_path(full, sigma=sigma, height=height)


def enforce_thickness(
    top: np.ndarray,
    bottom: np.ndarray,
    min_thickness: int,
    max_thickness: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    top_f = top.astype(np.float32)
    bottom_f = bottom.astype(np.float32)
    center = (top_f + bottom_f) / 2.0
    thickness = np.clip(bottom_f - top_f, float(min_thickness), float(max_thickness))
    top_f = center - thickness / 2.0
    bottom_f = center + thickness / 2.0
    top_i = np.clip(np.rint(top_f), 0, height - 2).astype(np.int32)
    bottom_i = np.clip(np.rint(bottom_f), 1, height - 1).astype(np.int32)

    too_thin = bottom_i - top_i < min_thickness
    if np.any(too_thin):
        bottom_i[too_thin] = np.minimum(height - 1, top_i[too_thin] + min_thickness)
        top_i[too_thin] = np.maximum(0, bottom_i[too_thin] - min_thickness)
    return top_i, bottom_i


def make_bright_mask(crop_gray: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float]:
    gray = crop_gray.astype(np.uint8, copy=False)
    work = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=1.4, sigmaY=0.8)
    threshold = float(np.percentile(work, 100.0 - args.bright_percent))
    threshold = max(threshold, float(np.mean(work) + args.mean_bias * np.std(work)))
    mask = work >= threshold

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(3, args.merge_x), max(3, args.merge_y)),
    )
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel, iterations=1)
    return mask_u8 > 0, work, threshold


def component_rows(labels: np.ndarray, label_id: int) -> np.ndarray:
    ys = np.where(labels == label_id)[0]
    return ys


def select_layer_mask(
    mask: np.ndarray,
    intensity: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    height, width = mask.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    candidates = []
    min_width = max(4, int(width * args.component_min_width_ratio))
    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < args.component_min_area or w < min_width:
            continue

        label_mask = labels == label_id
        mean_intensity = float(np.mean(intensity[label_mask]))
        cy = float(centroids[label_id][1])
        width_ratio = w / max(width, 1)
        area_ratio = area / max(width * height, 1)
        center_bias = math.exp(-0.5 * ((cy - height / 2.0) / max(height * 0.35, 1.0)) ** 2)
        border_penalty = 0.7 if y <= args.edge_margin or y + h >= height - args.edge_margin else 0.0
        score = 2.2 * width_ratio + 1.4 * area_ratio + mean_intensity / 255.0 + 0.45 * center_bias - border_penalty
        candidates.append(
            {
                "label": label_id,
                "x1": x,
                "x2": x + w - 1,
                "y1": y,
                "y2": y + h - 1,
                "cy": cy,
                "score": score,
            }
        )

    if not candidates:
        return mask

    seed = max(candidates, key=lambda item: item["score"])
    selected = [seed]
    y1 = seed["y1"]
    y2 = seed["y2"]
    changed = True
    while changed:
        changed = False
        for candidate in candidates:
            if candidate in selected:
                continue
            vertical_gap = max(candidate["y1"] - y2, y1 - candidate["y2"], 0)
            center_gap = abs(candidate["cy"] - ((y1 + y2) / 2.0))
            if vertical_gap <= args.merge_distance or center_gap <= args.merge_distance:
                selected.append(candidate)
                y1 = min(y1, candidate["y1"])
                y2 = max(y2, candidate["y2"])
                changed = True

    selected_mask = np.isin(labels, [item["label"] for item in selected])
    if np.count_nonzero(selected_mask) == 0:
        return mask
    return selected_mask


def mask_to_boundaries(
    selected_mask: np.ndarray,
    intensity: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, float]:
    height, width = selected_mask.shape
    columns = np.where(np.any(selected_mask, axis=0))[0]

    if len(columns) < 2:
        row_score = cv2.GaussianBlur(np.mean(intensity, axis=1).astype(np.float32)[None, :], (0, 0), sigmaX=4.0)[0]
        center = int(np.argmax(row_score))
        half = max(args.min_thickness // 2, args.band_height // 2)
        top = np.full(width, max(0, center - half), dtype=np.int32)
        bottom = np.full(width, min(height - 1, center + half), dtype=np.int32)
        return top, bottom, 0.0

    top_samples = []
    bottom_samples = []
    for x in columns:
        ys = np.where(selected_mask[:, x])[0]
        top_samples.append(int(ys.min()))
        bottom_samples.append(int(ys.max()))

    full_x = np.arange(width, dtype=np.float32)
    top = np.interp(full_x, columns.astype(np.float32), np.array(top_samples, dtype=np.float32))
    bottom = np.interp(full_x, columns.astype(np.float32), np.array(bottom_samples, dtype=np.float32))
    top = smooth_path(top, sigma=args.region_smooth, height=height)
    bottom = smooth_path(bottom, sigma=args.region_smooth, height=height)
    top, bottom = enforce_thickness(
        top,
        bottom,
        min_thickness=args.min_thickness,
        max_thickness=args.max_thickness,
        height=height,
    )

    values = [
        float(np.mean(intensity[top_y : bottom_y + 1, x]))
        for x, (top_y, bottom_y) in enumerate(zip(top, bottom))
    ]
    band_score = float(np.mean(values) / 255.0) if values else 0.0
    return top, bottom, band_score


def trace_layer(crop_gray: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float]:
    bright_mask, intensity, _ = make_bright_mask(crop_gray, args)
    selected_mask = select_layer_mask(bright_mask, intensity, args)
    return mask_to_boundaries(selected_mask, intensity, args)


def output_filename(row: dict) -> str:
    output_path = Path(row.get("output", ""))
    if output_path.name:
        name = output_path.name
        if name.endswith("_detect.png"):
            return name.replace("_detect.png", "_smas.png")
        return f"{output_path.stem}_smas.png"

    image_path = Path(row.get("image", ""))
    stem = image_path.stem
    if stem.startswith("frame_"):
        return f"frame__{stem[len('frame_'):]}_smas.png"
    return f"{stem}_smas.png"


def resolve_image_path(row: dict) -> Path:
    image_text = row.get("image", "")
    if image_text:
        image_path = Path(image_text)
        if image_path.exists():
            return image_path

    output_text = row.get("output", "")
    if output_text:
        output_path = Path(output_text)
        if output_path.exists():
            return output_path

    raise FileNotFoundError(f"Missing image for row: {row}")


def draw_result(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    top_path: np.ndarray | None,
    bottom_path: np.ndarray | None,
    band_score: float,
    line_width: int,
    draw_bbox: bool,
) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    left, top, right, bottom = bbox

    if draw_bbox:
        draw.rectangle([left, top, right - 1, bottom - 1], outline=(255, 140, 0, 180), width=2)

    if top_path is not None and bottom_path is not None and len(top_path) > 1:
        top_points = [(left + x, top + int(y)) for x, y in enumerate(top_path)]
        bottom_points = [(left + x, top + int(y)) for x, y in enumerate(bottom_path)]
        polygon = top_points + list(reversed(bottom_points))
        draw.polygon(polygon, fill=(255, 220, 0, 70))
        draw.line(top_points, fill=(0, 255, 255, 255), width=max(1, line_width))
        draw.line(bottom_points, fill=(255, 80, 255, 255), width=max(1, line_width))

        label = f"smas band={band_score:.3f}"
        label_y = max(0, top - 18)
        draw.rectangle([left, label_y, min(base.width - 1, left + 145), label_y + 16], fill=(0, 0, 0, 150))
        draw.text((left + 4, label_y + 2), label, fill=(255, 255, 255, 255))

    return Image.alpha_composite(base, overlay).convert("RGB")


def save_png(image: Image.Image, path: Path, png_compress: int) -> None:
    image.save(path, compress_level=max(0, min(9, int(png_compress))))


def trace_frame(
    row: dict,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    image_path = resolve_image_path(row)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    bbox = parse_bbox(row.get("bbox", ""))
    out_path = output_dir / output_filename(row)
    if bbox is None:
        save_png(image, out_path, args.png_compress)
        return {
            "image": str(image_path),
            "output": str(out_path),
            "bbox": "",
            "band_score": "",
            "upper_mean_y": "",
            "lower_mean_y": "",
            "mean_thickness": "",
            "angle_deg": "",
            "status": "missing_bbox",
        }

    clamped = clamp_bbox(bbox, width, height)
    if clamped is None:
        save_png(image, out_path, args.png_compress)
        return {
            "image": str(image_path),
            "output": str(out_path),
            "bbox": " ".join(f"{v:.2f}" for v in bbox),
            "band_score": "",
            "upper_mean_y": "",
            "lower_mean_y": "",
            "mean_thickness": "",
            "angle_deg": "",
            "status": "invalid_bbox",
        }

    left, top, right, bottom = clamped
    crop = np.array(image.crop((left, top, right, bottom)).convert("L"))
    top_path, bottom_path, band_score = trace_layer(crop, args)

    xs = np.arange(len(top_path), dtype=np.float32)
    midline = ((top_path + bottom_path) / 2.0).astype(np.float32)
    slope = float(np.polyfit(xs, midline, deg=1)[0]) if len(midline) > 1 else 0.0
    angle_deg = math.degrees(math.atan(slope))
    upper_mean_y = top + float(np.mean(top_path))
    lower_mean_y = top + float(np.mean(bottom_path))
    mean_thickness = float(np.mean(bottom_path - top_path))

    result = draw_result(
        image=image,
        bbox=clamped,
        top_path=top_path,
        bottom_path=bottom_path,
        band_score=band_score,
        line_width=args.line_width,
        draw_bbox=not args.no_bbox,
    )
    save_png(result, out_path, args.png_compress)
    return {
        "image": str(image_path),
        "output": str(out_path),
        "bbox": " ".join(f"{v:.2f}" for v in bbox),
        "band_score": f"{band_score:.6f}",
        "upper_mean_y": f"{upper_mean_y:.2f}",
        "lower_mean_y": f"{lower_mean_y:.2f}",
        "mean_thickness": f"{mean_thickness:.2f}",
        "angle_deg": f"{angle_deg:.3f}",
        "status": "ok",
    }


def read_prediction_rows(predictions_path: Path, max_images: int) -> list[dict]:
    with predictions_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda row: natural_key(Path(row.get("output", row.get("image", ""))).name))
    if max_images > 0:
        rows = rows[:max_images]
    return rows


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image",
        "output",
        "bbox",
        "band_score",
        "upper_mean_y",
        "lower_mean_y",
        "mean_thickness",
        "angle_deg",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace a bright SMAS layer inside detector bboxes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("output_id", nargs="?", default=None, help="Example: 25 resolves to output25.")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--bright-percent", type=float, default=35.0, help="Keep this brightest percent inside each bbox.")
    parser.add_argument("--mean-bias", type=float, default=0.0, help="Extra threshold as mean + bias * std.")
    parser.add_argument("--band-height", type=int, default=28, help="Fallback expected layer height in pixels.")
    parser.add_argument("--min-thickness", type=int, default=24, help="Minimum SMAS band thickness in pixels.")
    parser.add_argument("--max-thickness", type=int, default=64, help="Maximum SMAS band thickness in pixels.")
    parser.add_argument("--merge-x", type=int, default=81, help="Merge bright regions across horizontal gaps up to this scale.")
    parser.add_argument("--merge-y", type=int, default=13, help="Merge bright regions across vertical gaps up to this scale.")
    parser.add_argument("--merge-distance", type=int, default=20, help="Keep components near the selected layer in y.")
    parser.add_argument("--component-min-width-ratio", type=float, default=0.035, help="Minimum component width ratio.")
    parser.add_argument("--component-min-area", type=int, default=80, help="Minimum component area in pixels.")
    parser.add_argument("--edge-margin", type=int, default=3, help="Penalize components touching bbox top/bottom.")
    parser.add_argument("--region-smooth", type=float, default=32.0, help="Boundary smoothing sigma.")
    parser.add_argument("--line-width", type=int, default=2, help="Boundary line width in pixels.")
    parser.add_argument("--png-compress", type=int, default=1, help="PNG compression level. Lower is faster.")
    parser.add_argument("--max-images", type=int, default=0, help="0 means all frames.")
    parser.add_argument("--no-bbox", action="store_true", help="Do not draw the detector bbox.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_id = args.output_id or input("Output number (25 -> output25): ").strip() or "25"
    output_name = normalize_output_name(output_id)

    input_dir = Path(args.input_root).resolve() / output_name / "frames"
    output_dir = Path(args.output_root).resolve() / output_name / "frames"
    predictions_path = input_dir / "predictions.csv"

    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input folder: {input_dir}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing bbox prediction CSV: {predictions_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_prediction_rows(predictions_path, args.max_images)
    if not rows:
        raise RuntimeError(f"No prediction rows found in: {predictions_path}")

    print(f"[Input] {predictions_path}")
    print(f"[Output] {output_dir}")
    print(f"[Frames] {len(rows)}")

    summary = []
    for row in tqdm(rows, desc="trace-smas", unit="frame"):
        summary.append(trace_frame(row, output_dir, args))

    write_summary(output_dir / "smas_summary.csv", summary)
    ok_count = sum(row["status"] == "ok" for row in summary)
    print(f"[Done] traced {ok_count}/{len(summary)} frames")
    print(f"[Summary] {output_dir / 'smas_summary.csv'}")


if __name__ == "__main__":
    main()
