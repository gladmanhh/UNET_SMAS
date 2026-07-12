import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CLASSYS_ROOT = PROJECT_ROOT.parent / "CLASSYS-BEAUTY"
DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_RUN_DIR = BASE_DIR / "runs" / "full_32ch"
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "best.pt"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "outputs"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR
    RESAMPLE_NEAREST = Image.NEAREST


@dataclass(frozen=True)
class Size2D:
    width: int
    height: int


def normalize_output_name(value: str) -> str:
    text = str(value).strip()
    if text.lower().startswith("output"):
        suffix = text[6:]
        return f"output{int(suffix)}" if suffix.isdigit() else text
    if text.isdigit():
        return f"output{int(text)}"
    raise ValueError(f"Expected a number like 25 or a name like output25, got: {value}")


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def frame_output_name(image_path: Path, suffix: str) -> str:
    stem = image_path.stem
    if stem.startswith("frame_"):
        return f"frame__{stem[len('frame_'):]}_{suffix}.png"
    if stem.endswith("_detect"):
        return f"{stem[:-7]}_{suffix}.png"
    return f"{stem}_{suffix}.png"


def resolve_frames_dir(output_id: str, frames_dir: str | None = None) -> Path:
    if frames_dir:
        path = Path(frames_dir)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    output_name = normalize_output_name(output_id)
    candidates = [
        CLASSYS_ROOT / "data" / "frames" / output_name / "frames",
        PROJECT_ROOT / "data" / "frames" / output_name / "frames",
        PROJECT_ROOT / "outputs" / output_name / "frames",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Missing frame folder. Tried: " + ", ".join(str(p) for p in candidates))


def discover_frames(frame_dir: Path) -> list[Path]:
    frames = [p for p in frame_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    frames.sort(key=natural_key)
    if not frames:
        raise FileNotFoundError(f"No image frames found in: {frame_dir}")
    return frames


def tensor_from_image(image: Image.Image, size: Size2D) -> torch.Tensor:
    resized = image.resize((size.width, size.height), RESAMPLE_BILINEAR)
    arr = np.array(resized.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


def tensor_from_mask(mask: Image.Image, size: Size2D) -> torch.Tensor:
    resized = mask.resize((size.width, size.height), RESAMPLE_NEAREST)
    arr = (np.array(resized.convert("L")) > 0).astype(np.float32)
    return torch.from_numpy(arr[None, :, :])


def overlay_mask(image: Image.Image, mask: Image.Image, label: str = "PICOSAM2baseUNet") -> Image.Image:
    base = image.convert("RGBA")
    mask_arr = np.array(mask.convert("L")) > 0
    color = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    color[mask_arr] = [255, 220, 0, 95]
    overlay = Image.fromarray(color, mode="RGBA")
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([4, 4, min(image.width - 1, 236), 24], fill=(0, 0, 0, 150))
    draw.text((8, 8), label, fill=(255, 255, 255, 255))
    return Image.alpha_composite(base, overlay).convert("RGB")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image", "mask", "overlay", "mask_pixels", "mean_probability"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
