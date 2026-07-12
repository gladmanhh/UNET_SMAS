import argparse
import csv
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as VF
from tqdm import tqdm

from train_detector import build_model


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT.parent / "CLASSYS-BEAUTY"
DEFAULT_FRAME_ROOT = DEFAULT_SOURCE_ROOT / "data" / "frames"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"
DEFAULT_CHECKPOINT = SCRIPT_DIR / "runs" / "mobilenetv3_frcnn_8ep" / "best.pt"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PREDICTION_FIELDS = ["image", "output", "score", "bbox", "skipped"]


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def normalize_output_name(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("Output id is empty.")
    if text.lower().startswith("output"):
        suffix = text[6:]
        return f"output{int(suffix)}" if suffix.isdigit() else text
    if text.isdigit():
        return f"output{int(text)}"
    raise ValueError(f"Expected a number like 1 or an output name like output1, got: {value}")


def infer_output_name(input_dir: Path, explicit_output_id: str | None) -> str:
    if explicit_output_id:
        return normalize_output_name(explicit_output_id)
    if input_dir.name.lower() == "frames":
        return input_dir.parent.name
    return input_dir.name


def resolve_checkpoint(path_text: str | None) -> Path:
    if path_text:
        return Path(path_text).resolve()
    if DEFAULT_CHECKPOINT.exists():
        return DEFAULT_CHECKPOINT

    candidates = sorted(
        (SCRIPT_DIR / "runs").glob("*/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return DEFAULT_CHECKPOINT


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_detector(checkpoint_path: Path, device: torch.device, min_size: int | None, max_size: int | None):
    ckpt = load_checkpoint(checkpoint_path, device)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    model_min_size = min_size if min_size is not None else int(ckpt_args.get("min_size", 320))
    model_max_size = max_size if max_size is not None else int(ckpt_args.get("max_size", 640))

    model = build_model(
        num_classes=2,
        pretrained=False,
        min_size=model_min_size,
        max_size=model_max_size,
    ).to(device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model, model_min_size, model_max_size


def find_images(input_dir: Path, max_images: int) -> list[Path]:
    images = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_key,
    )
    if max_images > 0:
        return images[:max_images]
    return images


def output_filename(image_path: Path) -> str:
    stem = image_path.stem
    if stem.startswith("frame_"):
        return f"frame__{stem[len('frame_'):]}_detect.png"
    return f"{stem}_detect.png"


def clamp_box(box, width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def draw_detection(image: Image.Image, box: list[float] | None, score: float, score_thresh: float) -> Image.Image:
    output = image.copy()
    if box is None or score < score_thresh:
        return output

    draw = ImageDraw.Draw(output)
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=(230, 40, 40), width=3)

    label = f"smas {score:.3f}"
    try:
        label_box = draw.textbbox((0, 0), label)
        text_w = label_box[2] - label_box[0]
        text_h = label_box[3] - label_box[1]
    except AttributeError:
        text_w, text_h = draw.textsize(label)
    tx = max(0, int(x1))
    ty = max(0, int(y1) - text_h - 8)
    draw.rectangle([tx, ty, tx + text_w + 8, ty + text_h + 6], fill=(0, 0, 0))
    draw.text((tx + 4, ty + 3), label, fill=(255, 255, 255))
    return output


def read_predictions_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prediction_key(record: dict) -> str:
    return record.get("output") or record.get("image") or ""


def merge_prediction_records(existing: list[dict], current: list[dict]) -> list[dict]:
    merged = list(existing)
    index = {prediction_key(record): i for i, record in enumerate(merged) if prediction_key(record)}
    for record in current:
        key = prediction_key(record)
        if key in index:
            merged[index[key]] = record
        else:
            index[key] = len(merged)
            merged.append(record)
    return merged


@torch.no_grad()
def infer_images(
    model,
    images: list[Path],
    output_dir: Path,
    device: torch.device,
    score_thresh: float,
    skip_existing: bool,
    existing_by_output: dict[str, dict] | None = None,
    existing_by_image: dict[str, dict] | None = None,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_by_output = existing_by_output or {}
    existing_by_image = existing_by_image or {}
    records = []
    for image_path in tqdm(images, desc="infer", unit="image"):
        out_path = output_dir / output_filename(image_path)
        if skip_existing and out_path.exists():
            existing = existing_by_output.get(str(out_path)) or existing_by_image.get(str(image_path))
            if existing and existing.get("bbox", "").strip():
                record = existing.copy()
                record["image"] = str(image_path)
                record["output"] = str(out_path)
                record["skipped"] = "1"
                records.append(record)
                continue

        image = Image.open(image_path).convert("RGB")
        tensor = VF.to_tensor(image).to(device)
        prediction = model([tensor])[0]
        boxes = prediction["boxes"].detach().cpu()
        scores = prediction["scores"].detach().cpu()

        if len(scores) > 0:
            score = float(scores[0].item())
            box = clamp_box(boxes[0].tolist(), *image.size)
        else:
            score = 0.0
            box = None

        drawn = draw_detection(image, box, score, score_thresh)
        drawn.save(out_path)
        records.append(
            {
                "image": str(image_path),
                "output": str(out_path),
                "score": f"{score:.6f}",
                "bbox": "" if box is None else " ".join(f"{v:.2f}" for v in box),
                "skipped": "0",
            }
        )
    return records


def write_predictions_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SMAS bbox detection on CLASSYS-BEAUTY frame folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "output_id",
        nargs="?",
        default=None,
        help="Output number or name. Example: 1 resolves to output1.",
    )
    parser.add_argument("--input-root", default=str(DEFAULT_FRAME_ROOT))
    parser.add_argument("--input-dir", default=None, help="Exact frame folder. Overrides --input-root.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min-size", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=0, help="Limit images for a quick test. 0 means all.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_dir:
        input_dir = Path(args.input_dir).resolve()
        output_name = infer_output_name(input_dir, args.output_id)
    else:
        output_id = args.output_id or input("Output number (1 -> output1): ")
        output_name = normalize_output_name(output_id)
        input_dir = Path(args.input_root).resolve() / output_name / "frames"

    output_dir = Path(args.output_root).resolve() / output_name / "frames"
    checkpoint_path = resolve_checkpoint(args.checkpoint)

    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input frame folder: {input_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    images = find_images(input_dir, args.max_images)
    if not images:
        raise RuntimeError(f"No images found in: {input_dir}")

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("[Device] CUDA requested but unavailable. Falling back to CPU.")
        requested_device = torch.device("cpu")

    model, min_size, max_size = build_detector(
        checkpoint_path,
        requested_device,
        min_size=args.min_size,
        max_size=args.max_size,
    )
    print(f"[Checkpoint] {checkpoint_path}")
    print(f"[Device] {requested_device}")
    print(f"[Model] min_size={min_size} max_size={max_size} score_thresh={args.score_thresh}")
    print(f"[Input] {input_dir} ({len(images)} images)")
    print(f"[Output] {output_dir}")

    predictions_path = output_dir / "predictions.csv"
    existing_records = read_predictions_csv(predictions_path)
    existing_by_output = {
        record.get("output", ""): record for record in existing_records if record.get("output")
    }
    existing_by_image = {
        record.get("image", ""): record for record in existing_records if record.get("image")
    }
    records = infer_images(
        model=model,
        images=images,
        output_dir=output_dir,
        device=requested_device,
        score_thresh=args.score_thresh,
        skip_existing=args.skip_existing,
        existing_by_output=existing_by_output,
        existing_by_image=existing_by_image,
    )
    if existing_records and (args.skip_existing or args.max_images > 0):
        records_to_write = merge_prediction_records(existing_records, records)
    else:
        records_to_write = records
    write_predictions_csv(predictions_path, records_to_write)
    print(f"[Done] saved {len(records)} images to {output_dir}")


if __name__ == "__main__":
    main()
