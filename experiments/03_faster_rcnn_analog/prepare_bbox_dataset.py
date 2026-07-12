import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_SPLITS = {
    "train": ["output1", "output20", "output30", "output42"],
    "val": ["output10"],
    "test": ["output44"],
}


def parse_outputs(text: str) -> list[str]:
    outputs: list[str] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower().startswith("output"):
            outputs.append(f"output{int(token[6:])}")
        else:
            outputs.append(f"output{int(token)}")
    return outputs


def output_number(name: str) -> int:
    match = re.fullmatch(r"output(\d+)", name)
    if not match:
        raise ValueError(f"Invalid output name: {name}")
    return int(match.group(1))


def mask_to_bbox(mask_path: Path, pad: int = 0) -> tuple[list[int] | None, int, int, int]:
    mask = np.array(Image.open(mask_path).convert("L"))
    height, width = mask.shape[:2]
    ys, xs = np.where(mask > 0)
    area = int(xs.size)
    if area == 0:
        return None, area, width, height

    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(width, int(xs.max()) + 1 + pad)
    y2 = min(height, int(ys.max()) + 1 + pad)
    if x2 <= x1 or y2 <= y1:
        return None, 0, width, height
    return [x1, y1, x2, y2], area, width, height


def link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not overwrite:
            return "exists"
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
        return "copied"

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlinked"
        except OSError:
            shutil.copy2(src, dst)
            return "copied"

    raise ValueError(f"Unknown copy mode: {mode}")


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "split",
        "output",
        "frame",
        "image_path",
        "mask_path",
        "width",
        "height",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "mask_area",
        "has_box",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            bbox = row["bbox"]
            writer.writerow(
                {
                    "id": row["id"],
                    "split": row["split"],
                    "output": row["output"],
                    "frame": row["frame"],
                    "image_path": row["image_path"],
                    "mask_path": row["mask_path"],
                    "width": row["width"],
                    "height": row["height"],
                    "bbox_x1": "" if bbox is None else bbox[0],
                    "bbox_y1": "" if bbox is None else bbox[1],
                    "bbox_x2": "" if bbox is None else bbox[2],
                    "bbox_y2": "" if bbox is None else bbox[3],
                    "mask_area": row["mask_area"],
                    "has_box": int(bbox is not None),
                }
            )


def build_split_map(args: argparse.Namespace) -> dict[str, list[str]]:
    return {
        "train": parse_outputs(args.train_outputs),
        "val": parse_outputs(args.val_outputs),
        "test": parse_outputs(args.test_outputs),
    }


def prepare_dataset(args: argparse.Namespace) -> dict:
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    frame_root = source_root / "data" / "frames"
    label_root = source_root / "data" / "labeling_data"

    if not frame_root.exists():
        raise FileNotFoundError(f"Missing frame root: {frame_root}")
    if not label_root.exists():
        raise FileNotFoundError(f"Missing label root: {label_root}")

    split_map = build_split_map(args)
    available = {
        d.name
        for d in label_root.iterdir()
        if d.is_dir()
        and d.name.startswith("output")
        and args.min_output <= output_number(d.name) <= args.max_output
    }

    annotations: list[dict] = []
    skipped: list[dict] = []
    transfer_counts: dict[str, int] = {"exists": 0, "copied": 0, "hardlinked": 0}

    for split, outputs in split_map.items():
        for output_name in outputs:
            if output_name not in available:
                skipped.append(
                    {
                        "output": output_name,
                        "split": split,
                        "reason": "no_label_dir_in_requested_range",
                    }
                )
                continue

            ann_dir = label_root / output_name
            frame_dir = frame_root / output_name / "frames"
            if not frame_dir.exists():
                skipped.append(
                    {"output": output_name, "split": split, "reason": "missing_frame_dir"}
                )
                continue

            for mask_path in sorted(ann_dir.glob("*.png")):
                frame_path = frame_dir / mask_path.name
                if not frame_path.exists():
                    skipped.append(
                        {
                            "output": output_name,
                            "split": split,
                            "frame": mask_path.name,
                            "reason": "missing_frame",
                        }
                    )
                    continue

                sample_id = f"{output_name}_{mask_path.stem}"
                image_dst = output_root / "data" / "images" / split / f"{sample_id}.png"
                mask_dst = output_root / "data" / "masks" / split / f"{sample_id}.png"

                transfer_counts[link_or_copy(frame_path, image_dst, args.copy_mode, args.overwrite)] += 1
                transfer_counts[link_or_copy(mask_path, mask_dst, args.copy_mode, args.overwrite)] += 1

                bbox, mask_area, width, height = mask_to_bbox(mask_path, pad=args.bbox_pad)
                annotations.append(
                    {
                        "id": sample_id,
                        "split": split,
                        "output": output_name,
                        "frame": mask_path.name,
                        "image_path": image_dst.relative_to(output_root).as_posix(),
                        "mask_path": mask_dst.relative_to(output_root).as_posix(),
                        "width": width,
                        "height": height,
                        "bbox": bbox,
                        "mask_area": mask_area,
                    }
                )

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "requested_range": [args.min_output, args.max_output],
        "bbox_pad": args.bbox_pad,
        "copy_mode": args.copy_mode,
        "splits": split_map,
        "available_labeled_outputs_in_range": sorted(available, key=output_number),
        "transfer_counts": transfer_counts,
        "skipped": skipped,
        "counts": {},
    }

    for split in split_map:
        split_rows = [row for row in annotations if row["split"] == split]
        summary["counts"][split] = {
            "images": len(split_rows),
            "with_bbox": sum(row["bbox"] is not None for row in split_rows),
            "empty_mask": sum(row["bbox"] is None for row in split_rows),
            "outputs": sorted({row["output"] for row in split_rows}, key=output_number),
        }

    output_data = {
        "classes": [{"id": 1, "name": "smas"}],
        "annotations": annotations,
        "summary": summary,
    }

    write_json(output_root / "data" / "annotations.json", output_data)
    write_json(output_root / "data" / "summary.json", summary)
    for split in split_map:
        split_rows = [row for row in annotations if row["split"] == split]
        write_json(output_root / "data" / f"annotations_{split}.json", split_rows)
        write_csv(output_root / "data" / f"annotations_{split}.csv", split_rows)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare SMAS bbox detection data from binary masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-root",
        default=str(Path(__file__).resolve().parents[2].parent / "CLASSYS-BEAUTY"),
        help="Path to CLASSYS-BEAUTY.",
    )
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parent),
        help="Output folder for this experiment.",
    )
    parser.add_argument("--min-output", type=int, default=1)
    parser.add_argument("--max-output", type=int, default=44)
    parser.add_argument("--train-outputs", default=",".join(DEFAULT_SPLITS["train"]))
    parser.add_argument("--val-outputs", default=",".join(DEFAULT_SPLITS["val"]))
    parser.add_argument("--test-outputs", default=",".join(DEFAULT_SPLITS["test"]))
    parser.add_argument(
        "--copy-mode",
        choices=["hardlink", "copy"],
        default="hardlink",
        help="hardlink saves disk space and falls back to copy if needed.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bbox-pad", type=int, default=0, help="Optional bbox padding in pixels.")
    return parser.parse_args()


def main() -> None:
    summary = prepare_dataset(parse_args())
    print(json.dumps(summary["counts"], indent=2))
    if summary["skipped"]:
        print(f"Skipped entries: {len(summary['skipped'])}")
    print("Prepared:", summary["output_root"])


if __name__ == "__main__":
    main()
