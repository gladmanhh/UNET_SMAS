import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from common import (
    DEFAULT_OUTPUT_ROOT,
    RESAMPLE_NEAREST,
    Size2D,
    discover_frames,
    frame_output_name,
    normalize_output_name,
    resolve_frames_dir,
    tensor_from_image,
)
from model import PicoSAM2BaseUNet


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = BASE_DIR / "runs_multiclass" / "dermis_smas_32ch" / "best.pt"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "outputs_multiclass"
CLASS_NAMES = ["background", "dermis", "smas"]
COLORS = {
    1: (0, 220, 255, 90),
    2: (255, 220, 0, 110),
}


def load_model(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    class_names = ckpt_args.get("class_names", CLASS_NAMES)
    model = PicoSAM2BaseUNet(
        base_channels=int(ckpt_args.get("base_channels", 32)),
        out_channels=int(ckpt_args.get("num_classes", len(class_names))),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt_args


@torch.no_grad()
def predict_class_map(model: torch.nn.Module, image: Image.Image, image_size: Size2D, device: torch.device) -> Image.Image:
    tensor = tensor_from_image(image, image_size)[None, :, :, :].to(device)
    logits = model(tensor)
    pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    return Image.fromarray(pred, mode="L").resize(image.size, RESAMPLE_NEAREST)


def overlay_multiclass(image: Image.Image, class_map: Image.Image) -> Image.Image:
    base = image.convert("RGBA")
    arr = np.array(class_map, dtype=np.uint8)
    color = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    for class_id, rgba in COLORS.items():
        color[arr == class_id] = rgba
    overlay = Image.fromarray(color, mode="RGBA")
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([4, 4, min(image.width - 1, 310), 24], fill=(0, 0, 0, 150))
    draw.text((8, 8), "dermis=cyan, smas=yellow", fill=(255, 255, 255, 255))
    return Image.alpha_composite(base, overlay).convert("RGB")


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image", "class_mask", "dermis_mask", "smas_mask", "overlay", "dermis_pixels", "smas_pixels"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
    image_size = Size2D(int(ckpt_args.get("width", 384)), int(ckpt_args.get("height", 224)))

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (BASE_DIR / output_root).resolve()
    target_root = output_root / output_name
    frame_out = target_root / "frames"
    class_out = target_root / "class_masks"
    dermis_out = target_root / "dermis_masks"
    smas_out = target_root / "smas_masks"
    for folder in [frame_out, class_out, dermis_out, smas_out]:
        folder.mkdir(parents=True, exist_ok=True)

    print(f"[Input] {frame_dir}")
    print(f"[Checkpoint] {checkpoint}")
    print(f"[Output] {target_root}")
    print(f"[Frames] {len(frames)} device={device}")

    rows = []
    for frame_path in tqdm(frames, desc=f"multi-{output_name}", unit="frame"):
        image = Image.open(frame_path).convert("RGB")
        class_map = predict_class_map(model, image, image_size, device)
        arr = np.array(class_map, dtype=np.uint8)
        dermis = Image.fromarray(((arr == 1).astype(np.uint8) * 255), mode="L")
        smas = Image.fromarray(((arr == 2).astype(np.uint8) * 255), mode="L")

        class_path = class_out / frame_output_name(frame_path, "class")
        dermis_path = dermis_out / frame_output_name(frame_path, "dermis")
        smas_path = smas_out / frame_output_name(frame_path, "smas")
        overlay_path = frame_out / frame_output_name(frame_path, "multiclass")
        class_map.save(class_path, compress_level=1)
        dermis.save(dermis_path, compress_level=1)
        smas.save(smas_path, compress_level=1)
        overlay_multiclass(image, class_map).save(overlay_path, compress_level=1)
        rows.append(
            {
                "image": str(frame_path),
                "class_mask": str(class_path),
                "dermis_mask": str(dermis_path),
                "smas_mask": str(smas_path),
                "overlay": str(overlay_path),
                "dermis_pixels": int((arr == 1).sum()),
                "smas_pixels": int((arr == 2).sum()),
            }
        )

    save_csv(target_root / "predictions.csv", rows)
    print(f"[Done] {target_root}")
    return target_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer mutually exclusive dermis + SMAS masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("output_id", nargs="?", default=None)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--frames-dir", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.output_id is None:
        args.output_id = input("Output number (1 -> output1) [25]: ").strip() or "25"
    run_inference(args)


if __name__ == "__main__":
    main()
