import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as VF
from tqdm import tqdm

from common import DATA_ROOT, DEFAULT_RUN_DIR, PROJECT_ROOT, Size2D, tensor_from_image, tensor_from_mask
from model import PicoSAM2BaseUNet, count_params


class SMASFullFrameDataset(Dataset):
    def __init__(self, annotations: list[dict], split: str, image_size: Size2D, train: bool):
        self.rows = [row for row in annotations if row["split"] == split]
        if not self.rows:
            raise RuntimeError(f"No rows for split={split}")
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(PROJECT_ROOT / row["image_path"]).convert("RGB")
        mask = Image.open(PROJECT_ROOT / row["mask_path"]).convert("L")
        if self.train and np.random.random() < 0.5:
            image = VF.hflip(image)
            mask = VF.hflip(mask)
        return tensor_from_image(image, self.image_size), tensor_from_mask(mask, self.image_size), row["id"]


def load_annotations(path: Path = DATA_ROOT / "annotations.json") -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["annotations"]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    inter = torch.sum(probs * targets, dim=dims)
    denom = torch.sum(probs + targets, dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def seg_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pos = torch.clamp(targets.mean(), min=1e-4, max=0.5)
    pos_weight = torch.clamp((1.0 - pos) / pos, max=8.0).to(logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    return bce + dice_loss(logits, targets)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    dices = []
    ious = []
    for images, masks, _ in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        preds = torch.sigmoid(logits) >= 0.5
        targets = masks >= 0.5
        dims = tuple(range(1, preds.ndim))
        inter = torch.sum(preds & targets, dim=dims).float()
        pred_sum = torch.sum(preds, dim=dims).float()
        target_sum = torch.sum(targets, dim=dims).float()
        union = pred_sum + target_sum - inter
        dice = (2 * inter + 1e-6) / (pred_sum + target_sum + 1e-6)
        iou = (inter + 1e-6) / (union + 1e-6)
        valid = target_sum > 0
        if valid.any():
            dices.extend(dice[valid].detach().cpu().tolist())
            ious.extend(iou[valid].detach().cpu().tolist())
    return {
        "dice": float(np.mean(dices)) if dices else 0.0,
        "iou": float(np.mean(ious)) if ious else 0.0,
        "images": len(loader.dataset),
    }


def save_checkpoint(path: Path, model: torch.nn.Module, metrics: dict, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "metrics": metrics,
            "args": vars(args),
            "param_count": count_params(model),
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = Size2D(args.width, args.height)
    run_dir = Path(args.run_dir) if args.run_dir else DEFAULT_RUN_DIR
    annotations = load_annotations(Path(args.annotations))

    train_ds = SMASFullFrameDataset(annotations, "train", image_size=image_size, train=True)
    val_ds = SMASFullFrameDataset(annotations, "val", image_size=image_size, train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = PicoSAM2BaseUNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    best_iou = -1.0
    log_rows = []

    print(f"[Train] train={len(train_ds)} val={len(val_ds)}")
    print(f"[Train] params={count_params(model):,} device={device} run_dir={run_dir}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        losses = []
        for images, masks, _ in tqdm(train_loader, desc=f"train-{epoch}", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(images)
                loss = seg_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        metrics = evaluate(model, val_loader, device)
        metrics["train_loss"] = float(np.mean(losses))
        metrics["epoch"] = epoch
        metrics["seconds"] = time.perf_counter() - t0
        log_rows.append(metrics)
        print(
            f"Epoch {epoch:03d}/{args.epochs} loss={metrics['train_loss']:.4f} "
            f"val_iou={metrics['iou']:.4f} val_dice={metrics['dice']:.4f} {metrics['seconds']:.1f}s"
        )
        save_checkpoint(run_dir / "last.pt", model, metrics, args)
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            save_checkpoint(run_dir / "best.pt", model, metrics, args)

    summary = {
        "model": "PicoSAM2BaseUNet",
        "training": "full-frame image-only supervised segmentation",
        "param_count": count_params(model),
        "fp32_size_mb_estimate": count_params(model) * 4 / (1024 * 1024),
        "log": log_rows,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Done] {run_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train image-only PicoSAM2-style base U-Net for SMAS segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annotations", default=str(DATA_ROOT / "annotations.json"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
