import argparse
import csv
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as VF
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_annotations(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["annotations"]


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


class SMASBboxDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        annotations: list[dict],
        split: str,
        train: bool = False,
        include_empty: bool = True,
    ):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.train = train
        self.rows = [row for row in annotations if row["split"] == split]
        if not include_empty:
            self.rows = [row for row in self.rows if row["bbox"] is not None]
        if not self.rows:
            raise RuntimeError(f"No samples for split={split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = self.dataset_root / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        bbox = row["bbox"]
        if bbox is None:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            box_np = np.array([bbox], dtype=np.float32)
            if self.train and random.random() < 0.5:
                # Keep ultrasound vertical orientation fixed; mirror only left/right.
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                x1 = box_np[:, 0].copy()
                x2 = box_np[:, 2].copy()
                box_np[:, 0] = width - x2
                box_np[:, 2] = width - x1

            boxes = torch.as_tensor(box_np, dtype=torch.float32)
            labels = torch.ones((1,), dtype=torch.int64)
            area = torch.as_tensor(
                [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1])], dtype=torch.float32
            )
            iscrowd = torch.zeros((1,), dtype=torch.int64)

        if self.train:
            if random.random() < 0.4:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            if random.random() < 0.4:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.25))

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }
        return VF.to_tensor(image), target

    def row(self, index: int) -> dict:
        return self.rows[index]


def build_model(num_classes: int, pretrained: bool, min_size: int, max_size: int):
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT if pretrained else None
    try:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=weights,
            min_size=min_size,
            max_size=max_size,
        )
    except Exception as exc:
        if not pretrained:
            raise
        print(f"[Warning] pretrained detector weights unavailable: {exc}")
        print("[Warning] Falling back to randomly initialized detector.")
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None,
            weights_backbone=None,
            min_size=min_size,
            max_size=max_size,
        )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def box_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def average_precision(detections: list[dict], gt_boxes: dict[int, np.ndarray], iou_thr: float) -> float:
    total_gt = len(gt_boxes)
    if total_gt == 0:
        return float("nan")

    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    matched: set[int] = set()
    tp = []
    fp = []
    for det in detections:
        image_id = det["image_id"]
        gt_box = gt_boxes.get(image_id)
        iou = box_iou(det["box"], gt_box) if gt_box is not None else 0.0
        if gt_box is not None and iou >= iou_thr and image_id not in matched:
            matched.add(image_id)
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)

    if not tp:
        return 0.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(total_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    ap = 0.0
    for recall_level in np.linspace(0.0, 1.0, 101):
        keep = recalls >= recall_level
        ap += float(precisions[keep].max()) if keep.any() else 0.0
    return ap / 101.0


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    dataset: SMASBboxDataset,
    device: torch.device,
    score_thresh: float,
    min_ap_score: float,
) -> tuple[dict, list[dict]]:
    model.eval()
    records: list[dict] = []
    detections: list[dict] = []
    gt_boxes: dict[int, np.ndarray] = {}

    for images, targets in tqdm(loader, desc=f"eval-{dataset.split}", leave=False):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            idx = int(target["image_id"].item())
            row = dataset.row(idx)
            gt_box = None if row["bbox"] is None else np.array(row["bbox"], dtype=np.float32)
            if gt_box is not None:
                gt_boxes[idx] = gt_box

            pred_boxes = output["boxes"].detach().cpu().numpy()
            pred_scores = output["scores"].detach().cpu().numpy()
            for box, score in zip(pred_boxes, pred_scores):
                if float(score) >= min_ap_score:
                    detections.append(
                        {"image_id": idx, "box": box.astype(np.float32), "score": float(score)}
                    )

            if len(pred_scores) > 0:
                top_score = float(pred_scores[0])
                top_box = pred_boxes[0].astype(np.float32)
                top_iou = box_iou(top_box, gt_box)
                pred_box_list = [float(x) for x in top_box.tolist()]
            else:
                top_score = 0.0
                top_iou = 0.0
                pred_box_list = None

            records.append(
                {
                    "id": row["id"],
                    "output": row["output"],
                    "frame": row["frame"],
                    "split": dataset.split,
                    "has_gt": gt_box is not None,
                    "gt_bbox": None if gt_box is None else [float(x) for x in gt_box.tolist()],
                    "pred_score": top_score,
                    "pred_bbox": pred_box_list,
                    "top1_iou": top_iou,
                }
            )

    ap50 = average_precision(detections, gt_boxes, 0.5)
    ap_values = [
        average_precision(detections, gt_boxes, float(thr))
        for thr in np.arange(0.5, 1.0, 0.05)
    ]
    map50_95 = float(np.nanmean(ap_values)) if ap_values else float("nan")

    tp = fp = fn = 0
    fp_empty = 0
    gt_ious = []
    center_errors = []
    for rec in records:
        has_pred = rec["pred_score"] >= score_thresh
        has_gt = bool(rec["has_gt"])
        iou = float(rec["top1_iou"])
        if has_gt:
            gt_ious.append(iou)
            if rec["pred_bbox"] is not None:
                gb = np.array(rec["gt_bbox"], dtype=np.float32)
                pb = np.array(rec["pred_bbox"], dtype=np.float32)
                gcenter = np.array([(gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2])
                pcenter = np.array([(pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2])
                center_errors.append(float(np.linalg.norm(gcenter - pcenter)))
            if has_pred and iou >= 0.5:
                tp += 1
            else:
                fn += 1
                if has_pred:
                    fp += 1
        elif has_pred:
            fp += 1
            fp_empty += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    metrics = {
        "split": dataset.split,
        "images": len(records),
        "gt_images": int(sum(rec["has_gt"] for rec in records)),
        "empty_gt_images": int(sum(not rec["has_gt"] for rec in records)),
        "detections_for_ap": len(detections),
        "ap50": float(ap50),
        "map50_95": float(map50_95),
        "top1_mean_iou": float(np.mean(gt_ious)) if gt_ious else 0.0,
        "top1_median_iou": float(np.median(gt_ious)) if gt_ious else 0.0,
        "center_error_px_mean": float(np.mean(center_errors)) if center_errors else 0.0,
        "precision_score_0_5_iou_0_5": float(precision),
        "recall_score_0_5_iou_0_5": float(recall),
        "false_positive_empty_score_0_5": int(fp_empty),
        "score_threshold": score_thresh,
        "min_ap_score": min_ap_score,
    }
    return metrics, records


def save_prediction_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "split",
        "output",
        "frame",
        "has_gt",
        "gt_bbox",
        "pred_score",
        "pred_bbox",
        "top1_iou",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            row = rec.copy()
            row["gt_bbox"] = "" if rec["gt_bbox"] is None else " ".join(f"{x:.2f}" for x in rec["gt_bbox"])
            row["pred_bbox"] = "" if rec["pred_bbox"] is None else " ".join(f"{x:.2f}" for x in rec["pred_bbox"])
            writer.writerow(row)


def save_visualizations(
    dataset_root: Path,
    records: list[dict],
    annotations: list[dict],
    output_dir: Path,
    max_images: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row_by_id = {row["id"]: row for row in annotations}
    records_sorted = sorted(records, key=lambda r: (r["output"], r["frame"]))
    for rec in records_sorted[:max_images]:
        row = row_by_id[rec["id"]]
        image = Image.open(dataset_root / row["image_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        if rec["gt_bbox"] is not None:
            x1, y1, x2, y2 = rec["gt_bbox"]
            draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=(0, 220, 0), width=3)
        if rec["pred_bbox"] is not None and rec["pred_score"] > 0:
            x1, y1, x2, y2 = rec["pred_bbox"]
            draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=(230, 40, 40), width=3)
        label = f"{rec['id']} score={rec['pred_score']:.3f} iou={rec['top1_iou']:.3f}"
        draw.rectangle([4, 4, 620, 30], fill=(0, 0, 0))
        draw.text((8, 10), label, fill=(255, 255, 255))
        image.save(output_dir / f"{rec['id']}.png")


@torch.no_grad()
def benchmark(model, dataset: SMASBboxDataset, device: torch.device, max_images: int) -> dict:
    model.eval()
    count = min(len(dataset), max_images)
    if count == 0:
        return {"images": 0, "seconds": 0.0, "ms_per_image": 0.0, "fps": 0.0}

    images = [dataset[i][0].to(device) for i in range(count)]
    warmup = min(10, count)
    for i in range(warmup):
        _ = model([images[i]])
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for img in images:
        _ = model([img])
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "images": count,
        "seconds": elapsed,
        "ms_per_image": 1000.0 * elapsed / count,
        "fps": count / max(elapsed, 1e-12),
    }


def train_one_epoch(model, loader, optimizer, device, epoch: int, amp: bool) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    for images, targets in tqdm(loader, desc=f"train-{epoch}", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().cpu())
        num_batches += 1

    return total_loss / max(num_batches, 1)


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "metrics": metrics,
            "args": vars(args),
            "classes": {"background": 0, "smas": 1},
        },
        path,
    )


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight SMAS bbox detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        default=str(Path(__file__).resolve().parent),
        help="Experiment root containing data/annotations.json and data/images.",
    )
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--min-size", type=int, default=320)
    parser.add_argument("--max-size", type=int, default=640)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--min-ap-score", type=float, default=0.001)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--drop-empty-train", action="store_true")
    parser.add_argument("--visualize", type=int, default=24)
    parser.add_argument("--benchmark-images", type=int, default=100)
    parser.add_argument("--eval-only", default=None, help="Path to checkpoint for evaluation only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_root = Path(args.dataset_root).resolve()
    annotations_path = (
        Path(args.annotations).resolve()
        if args.annotations
        else dataset_root / "data" / "annotations.json"
    )
    annotations = load_annotations(annotations_path)

    run_dir = (
        Path(args.run_dir).resolve()
        if args.run_dir
        else Path(__file__).resolve().parent / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "args.json", vars(args))

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[Device] {device}")

    train_ds = SMASBboxDataset(
        dataset_root,
        annotations,
        "train",
        train=True,
        include_empty=not args.drop_empty_train,
    )
    val_ds = SMASBboxDataset(dataset_root, annotations, "val", train=False, include_empty=True)
    test_ds = SMASBboxDataset(dataset_root, annotations, "test", train=False, include_empty=True)
    print(f"[Data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        num_classes=2,
        pretrained=not args.no_pretrained,
        min_size=args.min_size,
        max_size=args.max_size,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] params={total_params:,} trainable={trainable_params:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    if args.eval_only:
        ckpt = torch.load(args.eval_only, map_location=device)
        model.load_state_dict(ckpt["model"])
        best_epoch = int(ckpt.get("epoch", -1))
        best_metric = ckpt.get("metrics", {})
        print(f"[EvalOnly] loaded epoch={best_epoch} metrics={best_metric}")
    else:
        log_path = run_dir / "train_log.csv"
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_ap50",
                    "val_map50_95",
                    "val_top1_mean_iou",
                    "val_precision",
                    "val_recall",
                    "lr",
                    "seconds",
                ]
            )

        best_ap50 = -math.inf
        for epoch in range(1, args.epochs + 1):
            t0 = time.perf_counter()
            train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args.amp)
            scheduler.step()
            val_metrics, val_records = evaluate(
                model, val_loader, val_ds, device, args.score_thresh, args.min_ap_score
            )
            elapsed = time.perf_counter() - t0
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"loss={train_loss:.4f} "
                f"val_ap50={val_metrics['ap50']:.4f} "
                f"val_map={val_metrics['map50_95']:.4f} "
                f"val_iou={val_metrics['top1_mean_iou']:.4f} "
                f"{elapsed:.1f}s"
            )
            with log_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch,
                        train_loss,
                        val_metrics["ap50"],
                        val_metrics["map50_95"],
                        val_metrics["top1_mean_iou"],
                        val_metrics["precision_score_0_5_iou_0_5"],
                        val_metrics["recall_score_0_5_iou_0_5"],
                        lr,
                        elapsed,
                    ]
                )
            save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, val_metrics, args)
            if val_metrics["ap50"] > best_ap50:
                best_ap50 = val_metrics["ap50"]
                save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, val_metrics, args)
                save_prediction_csv(run_dir / "predictions_val_best.csv", val_records)

        ckpt = torch.load(run_dir / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[Best] epoch={ckpt['epoch']} val_ap50={ckpt['metrics']['ap50']:.4f}")

    val_metrics, val_records = evaluate(
        model, val_loader, val_ds, device, args.score_thresh, args.min_ap_score
    )
    test_metrics, test_records = evaluate(
        model, test_loader, test_ds, device, args.score_thresh, args.min_ap_score
    )
    speed = benchmark(model, test_ds, device, args.benchmark_images)
    final_metrics = {
        "model": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "parameter_count": total_params,
        "trainable_parameter_count": trainable_params,
        "val": val_metrics,
        "test": test_metrics,
        "test_inference_speed_batch1": speed,
    }
    write_json(run_dir / "final_metrics.json", final_metrics)
    save_prediction_csv(run_dir / "predictions_val.csv", val_records)
    save_prediction_csv(run_dir / "predictions_test.csv", test_records)
    save_visualizations(
        dataset_root,
        test_records,
        annotations,
        run_dir / "visualizations" / "test",
        args.visualize,
    )
    print(json.dumps(final_metrics, indent=2))
    print(f"[Done] {run_dir}")


if __name__ == "__main__":
    main()
