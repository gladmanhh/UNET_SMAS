import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT_ROOT,
    Size2D,
    discover_frames,
    frame_output_name,
    normalize_output_name,
    overlay_mask,
    resolve_frames_dir,
    tensor_from_image,
    write_csv,
    RESAMPLE_NEAREST,
)
from model import PicoSAM2BaseUNet


def load_model(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    model = PicoSAM2BaseUNet(base_channels=int(ckpt_args.get("base_channels", 32))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt_args


@torch.no_grad()
def predict_mask(
    model: torch.nn.Module,
    image: Image.Image,
    image_size: Size2D,
    device: torch.device,
    threshold: float,
) -> tuple[Image.Image, int, float]:
    tensor = tensor_from_image(image, image_size)[None, :, :, :].to(device)
    logits = model(tensor)
    probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    mask = (probs >= threshold).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask, mode="L").resize(image.size, RESAMPLE_NEAREST)
    mask_pixels = int((np.array(mask_img) > 0).sum())
    mean_probability = float(probs.mean())
    return mask_img, mask_pixels, mean_probability


def run_inference(args: argparse.Namespace) -> Path:
    output_name = normalize_output_name(args.output_id)
    frame_dir = resolve_frames_dir(output_name, args.frames_dir)
    frames = discover_frames(frame_dir)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (Path(__file__).resolve().parent / checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_args = load_model(checkpoint, device)
    image_size = Size2D(int(ckpt_args.get("width", ckpt_args.get("full_width", 384))), int(ckpt_args.get("height", ckpt_args.get("full_height", 224))))
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (Path(__file__).resolve().parent / output_root).resolve()
    target_root = output_root / output_name
    frame_out = target_root / "frames"
    mask_out = target_root / "masks"
    frame_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    print(f"[Input] {frame_dir}")
    print(f"[Checkpoint] {checkpoint}")
    print(f"[Output] {target_root}")
    print(f"[Frames] {len(frames)} device={device}")

    rows = []
    for frame_path in tqdm(frames, desc=f"picosam2base-{output_name}", unit="frame"):
        image = Image.open(frame_path).convert("RGB")
        mask, mask_pixels, mean_probability = predict_mask(model, image, image_size, device, args.threshold)
        mask_path = mask_out / frame_output_name(frame_path, "mask")
        overlay_path = frame_out / frame_output_name(frame_path, "seg")
        mask.save(mask_path, compress_level=1)
        overlay_mask(image, mask).save(overlay_path, compress_level=1)
        rows.append(
            {
                "image": str(frame_path),
                "mask": str(mask_path),
                "overlay": str(overlay_path),
                "mask_pixels": mask_pixels,
                "mean_probability": f"{mean_probability:.6f}",
            }
        )

    write_csv(target_root / "predictions.csv", rows)
    print(f"[Done] {target_root}")
    return target_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer SMAS masks with image-only PicoSAM2BaseUNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("output_id", nargs="?", default=None, help="Example: 25 resolves to output25.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--frames-dir", default=None, help="Optional direct frame folder.")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.output_id is None:
        args.output_id = input("Output number (1 -> output1) [25]: ").strip() or "25"
    run_inference(args)


if __name__ == "__main__":
    main()
