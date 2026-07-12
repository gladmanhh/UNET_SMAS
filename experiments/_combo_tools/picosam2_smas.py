import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as VF
from tqdm import tqdm


TOOL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOL_DIR.parents[1]
SCRIPT_DIR = PROJECT_ROOT
CLASSYS_ROOT = PROJECT_ROOT.parent / "CLASSYS-BEAUTY"
ANNOTATIONS_PATH = PROJECT_ROOT / "data" / "annotations.json"
PICOSAM_ROOT = TOOL_DIR / "runs_picosam2"
TINYBOX_ROOT = TOOL_DIR / "runs_tinybox"
EXPERIMENT_ROOT = TOOL_DIR / "outputs_picosam2"
FRCNN_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "03_faster_rcnn_analog" / "outputs"


@dataclass
class Size2D:
    width: int
    height: int


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_annotations(path: Path = ANNOTATIONS_PATH) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["annotations"]


def normalize_output_name(value: str) -> str:
    text = value.strip()
    if text.lower().startswith("output"):
        suffix = text[6:]
        return f"output{int(suffix)}" if suffix.isdigit() else text
    if text.isdigit():
        return f"output{int(text)}"
    raise ValueError(f"Expected a number like 25 or name like output25, got: {value}")


def natural_key(path: Path) -> list[object]:
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def frame_output_name(image_path: Path, suffix: str) -> str:
    stem = image_path.stem
    if stem.startswith("frame_"):
        return f"frame__{stem[len('frame_'):]}_{suffix}.png"
    if stem.endswith("_detect"):
        return f"{stem[:-7]}_{suffix}.png"
    return f"{stem}_{suffix}.png"


def mask_to_bbox(mask: Image.Image) -> list[float] | None:
    arr = np.array(mask.convert("L"))
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def clamp_bbox(box: list[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    left = max(0, min(width - 1, int(math.floor(x1))))
    top = max(0, min(height - 1, int(math.floor(y1))))
    right = max(0, min(width, int(math.ceil(x2))))
    bottom = max(0, min(height, int(math.ceil(y2))))
    if right <= left + 1 or bottom <= top + 1:
        return None
    return left, top, right, bottom


def jitter_bbox(
    box: list[float],
    width: int,
    height: int,
    scale: float,
    rng: np.random.Generator,
) -> list[float]:
    if scale <= 0:
        return box
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    dx1 = rng.uniform(-scale, scale) * bw
    dx2 = rng.uniform(-scale, scale) * bw
    dy1 = rng.uniform(-scale, scale) * bh
    dy2 = rng.uniform(-scale, scale) * bh
    return [
        max(0.0, x1 + dx1),
        max(0.0, y1 + dy1),
        min(float(width), x2 + dx2),
        min(float(height), y2 + dy2),
    ]


class DSConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvPair(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            DSConv(in_ch, out_ch),
            DSConv(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class PicoSAM2UNet(nn.Module):
    """Depthwise-separable U-Net inspired by PicoSAM2's edge-friendly design."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32, out_channels: int = 1):
        super().__init__()
        c1 = base_channels
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8
        c5 = c1 * 16

        self.enc1 = ConvPair(in_channels, c1)
        self.enc2 = ConvPair(c1, c2)
        self.enc3 = ConvPair(c2, c3)
        self.enc4 = ConvPair(c3, c4)
        self.bottleneck = ConvPair(c4, c5)

        self.up4 = ConvPair(c5 + c4, c4)
        self.up3 = ConvPair(c4 + c3, c3)
        self.up2 = ConvPair(c3 + c2, c2)
        self.up1 = ConvPair(c2 + c1, c1)
        self.head = nn.Conv2d(c1, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))

        x = F.interpolate(b, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up4(torch.cat([x, e4], dim=1))
        x = F.interpolate(x, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(torch.cat([x, e3], dim=1))
        x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, e2], dim=1))
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, e1], dim=1))
        return self.head(x)


class TinyBoxNet(nn.Module):
    def __init__(self, base_channels: int = 16):
        super().__init__()
        c1 = base_channels
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 6
        c5 = c1 * 8
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            DSConv(c1, c2, stride=2),
            DSConv(c2, c3, stride=2),
            DSConv(c3, c4, stride=2),
            DSConv(c4, c5, stride=2),
            DSConv(c5, c5, stride=1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c5, c5),
            nn.ReLU(inplace=True),
            nn.Linear(c5, 5),
        )

    def forward(self, x):
        return self.head(self.features(x))


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def tensor_from_image(image: Image.Image, size: Size2D) -> torch.Tensor:
    resized = image.resize((size.width, size.height), Image.BILINEAR)
    return VF.to_tensor(resized)


def tensor_from_mask(mask: Image.Image, size: Size2D) -> torch.Tensor:
    resized = mask.resize((size.width, size.height), Image.NEAREST)
    arr = (np.array(resized.convert("L")) > 0).astype(np.float32)
    return torch.from_numpy(arr[None, :, :])


class SMASSegDataset(Dataset):
    def __init__(
        self,
        annotations: list[dict],
        split: str,
        mode: str,
        image_size: Size2D,
        crop_size: Size2D,
        train: bool = False,
        jitter: float = 0.08,
    ):
        self.rows = [row for row in annotations if row["split"] == split]
        if mode == "crop":
            self.rows = [row for row in self.rows if row["bbox"] is not None]
        if not self.rows:
            raise RuntimeError(f"No rows for split={split} mode={mode}")
        self.mode = mode
        self.image_size = image_size
        self.crop_size = crop_size
        self.train = train
        self.jitter = jitter
        self.rng = np.random.default_rng(42 if not train else None)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(SCRIPT_DIR / row["image_path"]).convert("RGB")
        mask = Image.open(SCRIPT_DIR / row["mask_path"]).convert("L")

        if self.mode == "full":
            if self.train and np.random.random() < 0.5:
                image = VF.hflip(image)
                mask = VF.hflip(mask)
            return tensor_from_image(image, self.image_size), tensor_from_mask(mask, self.image_size), row["id"]

        bbox = row["bbox"]
        if self.train:
            bbox = jitter_bbox(bbox, image.width, image.height, self.jitter, self.rng)
        clamped = clamp_bbox(bbox, image.width, image.height)
        if clamped is None:
            clamped = (0, 0, image.width, image.height)
        image_crop = image.crop(clamped)
        mask_crop = mask.crop(clamped)
        if self.train and np.random.random() < 0.5:
            image_crop = VF.hflip(image_crop)
            mask_crop = VF.hflip(mask_crop)
        return (
            tensor_from_image(image_crop, self.crop_size),
            tensor_from_mask(mask_crop, self.crop_size),
            row["id"],
        )


class TinyBoxDataset(Dataset):
    def __init__(self, annotations: list[dict], split: str, input_size: Size2D, train: bool = False):
        self.rows = [row for row in annotations if row["split"] == split]
        if not self.rows:
            raise RuntimeError(f"No rows for split={split}")
        self.input_size = input_size
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(SCRIPT_DIR / row["image_path"]).convert("RGB")
        width, height = image.size
        bbox = row["bbox"]
        if self.train and np.random.random() < 0.5:
            image = VF.hflip(image)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                bbox = [width - x2, y1, width - x1, y2]

        image_tensor = tensor_from_image(image, self.input_size)
        obj = 0.0 if bbox is None else 1.0
        target = torch.zeros(5, dtype=torch.float32)
        target[0] = obj
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            target[1:] = torch.tensor([x1 / width, y1 / height, x2 / width, y2 / height], dtype=torch.float32)
        return image_tensor, target, row["id"], torch.tensor([width, height], dtype=torch.float32)


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
def evaluate_seg(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    dices = []
    ious = []
    for images, masks, _ in tqdm(loader, desc="eval-seg", leave=False):
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


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_checkpoint(path: Path, model: nn.Module, metrics: dict, args: argparse.Namespace) -> None:
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


def train_segmentation(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    annotations = load_annotations()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = Size2D(args.full_width, args.full_height)
    crop_size = Size2D(args.crop_width, args.crop_height)
    run_dir = Path(args.run_dir) if args.run_dir else PICOSAM_ROOT / f"{args.mode}_{args.base_channels}ch"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SMASSegDataset(
        annotations,
        "train",
        mode=args.mode,
        image_size=image_size,
        crop_size=crop_size,
        train=True,
        jitter=args.jitter,
    )
    val_ds = SMASSegDataset(
        annotations,
        "val",
        mode=args.mode,
        image_size=image_size,
        crop_size=crop_size,
        train=False,
    )
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

    model = PicoSAM2UNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    best_iou = -1.0
    log_rows = []
    print(f"[SegTrain] mode={args.mode} train={len(train_ds)} val={len(val_ds)}")
    print(f"[SegTrain] params={count_params(model):,} device={device} run_dir={run_dir}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        losses = []
        for images, masks, _ in tqdm(train_loader, desc=f"seg-train-{epoch}", leave=False):
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

        metrics = evaluate_seg(model, val_loader, device)
        metrics["train_loss"] = float(np.mean(losses))
        metrics["epoch"] = epoch
        metrics["seconds"] = time.perf_counter() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs} loss={metrics['train_loss']:.4f} "
            f"val_iou={metrics['iou']:.4f} val_dice={metrics['dice']:.4f} {metrics['seconds']:.1f}s"
        )
        log_rows.append(metrics)
        save_checkpoint(run_dir / "last.pt", model, metrics, args)
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            save_checkpoint(run_dir / "best.pt", model, metrics, args)

    save_json(
        run_dir / "final_metrics.json",
        {
            "mode": args.mode,
            "param_count": count_params(model),
            "fp32_size_mb_estimate": count_params(model) * 4 / (1024 * 1024),
            "log": log_rows,
        },
    )
    print(f"[Done] {run_dir}")


@torch.no_grad()
def evaluate_box(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    ious = []
    obj_correct = 0
    total = 0
    for images, targets, _, _ in tqdm(loader, desc="eval-box", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        raw = model(images)
        obj = torch.sigmoid(raw[:, 0])
        boxes = torch.sigmoid(raw[:, 1:])
        gt_obj = targets[:, 0]
        obj_correct += int(((obj >= 0.5).float() == gt_obj).sum().item())
        total += int(len(images))
        for pred_box, target in zip(boxes.detach().cpu(), targets.detach().cpu()):
            if target[0] < 0.5:
                continue
            ious.append(float(box_iou_np(pred_box.numpy(), target[1:].numpy())))
    return {
        "images": total,
        "objectness_acc": obj_correct / max(total, 1),
        "box_iou": float(np.mean(ious)) if ious else 0.0,
    }


def box_iou_np(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def train_tinybox(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    annotations = load_annotations()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = Size2D(args.width, args.height)
    run_dir = Path(args.run_dir) if args.run_dir else TINYBOX_ROOT / f"tinybox_{args.base_channels}ch"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = TinyBoxDataset(annotations, "train", input_size=input_size, train=True)
    val_ds = TinyBoxDataset(annotations, "val", input_size=input_size, train=False)
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
    model = TinyBoxNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_iou = -1.0
    log_rows = []
    print(f"[BoxTrain] train={len(train_ds)} val={len(val_ds)} params={count_params(model):,} device={device}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        losses = []
        for images, targets, _, _ in tqdm(train_loader, desc=f"box-train-{epoch}", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            raw = model(images)
            obj_loss = F.binary_cross_entropy_with_logits(raw[:, 0], targets[:, 0])
            pred_box = torch.sigmoid(raw[:, 1:])
            pos = targets[:, 0] > 0.5
            if pos.any():
                box_loss = F.smooth_l1_loss(pred_box[pos], targets[pos, 1:])
            else:
                box_loss = raw[:, 1:].sum() * 0.0
            loss = obj_loss + args.box_loss_weight * box_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        metrics = evaluate_box(model, val_loader, device)
        metrics["train_loss"] = float(np.mean(losses))
        metrics["epoch"] = epoch
        metrics["seconds"] = time.perf_counter() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs} loss={metrics['train_loss']:.4f} "
            f"val_box_iou={metrics['box_iou']:.4f} obj_acc={metrics['objectness_acc']:.4f} "
            f"{metrics['seconds']:.1f}s"
        )
        log_rows.append(metrics)
        save_checkpoint(run_dir / "last.pt", model, metrics, args)
        if metrics["box_iou"] > best_iou:
            best_iou = metrics["box_iou"]
            save_checkpoint(run_dir / "best.pt", model, metrics, args)

    save_json(
        run_dir / "final_metrics.json",
        {
            "param_count": count_params(model),
            "fp32_size_mb_estimate": count_params(model) * 4 / (1024 * 1024),
            "log": log_rows,
        },
    )
    print(f"[Done] {run_dir}")


def load_picosam(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    model = PicoSAM2UNet(base_channels=int(ckpt_args.get("base_channels", 32))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt_args


def load_tinybox(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    model = TinyBoxNet(base_channels=int(ckpt_args.get("base_channels", 16))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt_args


def predict_crop_mask(
    model: nn.Module,
    image: Image.Image,
    bbox: list[float],
    crop_size: Size2D,
    device: torch.device,
    threshold: float,
) -> Image.Image:
    clamped = clamp_bbox(bbox, image.width, image.height)
    mask_full = Image.new("L", image.size, 0)
    if clamped is None:
        return mask_full
    crop = image.crop(clamped)
    tensor = tensor_from_image(crop, crop_size)[None, :, :, :].to(device)
    logits = model(tensor)
    probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    mask = (probs >= threshold).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask, mode="L").resize((clamped[2] - clamped[0], clamped[3] - clamped[1]), Image.NEAREST)
    mask_full.paste(mask_img, (clamped[0], clamped[1]))
    return mask_full


def predict_full_mask(
    model: nn.Module,
    image: Image.Image,
    image_size: Size2D,
    device: torch.device,
    threshold: float,
) -> Image.Image:
    tensor = tensor_from_image(image, image_size)[None, :, :, :].to(device)
    logits = model(tensor)
    probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    mask = (probs >= threshold).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L").resize(image.size, Image.NEAREST)


def predict_tinybox(model: nn.Module, image: Image.Image, input_size: Size2D, device: torch.device) -> tuple[float, list[float]]:
    tensor = tensor_from_image(image, input_size)[None, :, :, :].to(device)
    raw = model(tensor)[0]
    score = float(torch.sigmoid(raw[0]).detach().cpu())
    box = torch.sigmoid(raw[1:]).detach().cpu().numpy().astype(float)
    x1, y1, x2, y2 = box.tolist()
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return score, [x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height]


def overlay_mask(image: Image.Image, mask: Image.Image, bbox: list[float] | None, label: str) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask_arr = np.array(mask.convert("L")) > 0
    color = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    color[mask_arr] = [255, 220, 0, 95]
    overlay = Image.alpha_composite(overlay, Image.fromarray(color, mode="RGBA"))
    draw = ImageDraw.Draw(overlay)
    if bbox is not None:
        clamped = clamp_bbox(bbox, image.width, image.height)
        if clamped is not None:
            draw.rectangle([clamped[0], clamped[1], clamped[2] - 1, clamped[3] - 1], outline=(0, 255, 255, 220), width=2)
    draw.rectangle([4, 4, min(image.width - 1, 340), 24], fill=(0, 0, 0, 150))
    draw.text((8, 8), label, fill=(255, 255, 255, 255))
    return Image.alpha_composite(base, overlay).convert("RGB")


def read_frcnn_predictions(output_name: str) -> dict[str, dict]:
    path = FRCNN_OUTPUT_ROOT / output_name / "frames" / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Faster R-CNN predictions: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {Path(row["image"]).name: row for row in rows}


def parse_bbox(text: str) -> list[float] | None:
    parts = text.split()
    if len(parts) != 4:
        return None
    return [float(x) for x in parts]


def save_experiment_record(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image", "mask", "overlay", "bbox", "score"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def run_output_experiments(args: argparse.Namespace) -> None:
    output_name = normalize_output_name(args.output_id)
    frame_dir = CLASSYS_ROOT / "data" / "frames" / output_name / "frames"
    if not frame_dir.exists():
        raise FileNotFoundError(f"Missing frame folder: {frame_dir}")
    frames = sorted([p for p in frame_dir.glob("*.png")], key=natural_key)
    if args.max_images > 0:
        frames = frames[: args.max_images]
    if not frames:
        raise RuntimeError(f"No frames found in {frame_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crop_model, crop_args = load_picosam(Path(args.crop_checkpoint), device)
    full_model, full_args = load_picosam(Path(args.full_checkpoint), device)
    tiny_model, tiny_args = load_tinybox(Path(args.tinybox_checkpoint), device)
    crop_size = Size2D(int(crop_args.get("crop_width", 256)), int(crop_args.get("crop_height", 128)))
    full_size = Size2D(int(full_args.get("full_width", 384)), int(full_args.get("full_height", 224)))
    tiny_size = Size2D(int(tiny_args.get("width", 320)), int(tiny_args.get("height", 192)))
    frcnn_rows = read_frcnn_predictions(output_name)

    combos = [
        "01_picosam2_fasterrcnn",
        "02_picosam2_only",
        "03_picosam2_tinybox",
    ]
    root = Path(args.output_root) / output_name
    for combo in combos:
        (root / combo / "frames").mkdir(parents=True, exist_ok=True)
        (root / combo / "masks").mkdir(parents=True, exist_ok=True)

    records = {combo: [] for combo in combos}
    for frame_path in tqdm(frames, desc=f"picosam2-{output_name}", unit="frame"):
        image = Image.open(frame_path).convert("RGB")

        frcnn_row = frcnn_rows.get(frame_path.name)
        frcnn_bbox = parse_bbox(frcnn_row.get("bbox", "")) if frcnn_row else None
        if frcnn_bbox is None:
            frcnn_mask = Image.new("L", image.size, 0)
        else:
            frcnn_mask = predict_crop_mask(crop_model, image, frcnn_bbox, crop_size, device, args.threshold)
        save_combo_result(root, "01_picosam2_fasterrcnn", frame_path, image, frcnn_mask, frcnn_bbox, records, frcnn_row.get("score", "") if frcnn_row else "")

        full_mask = predict_full_mask(full_model, image, full_size, device, args.threshold)
        save_combo_result(root, "02_picosam2_only", frame_path, image, full_mask, None, records, "")

        tiny_score, tiny_bbox = predict_tinybox(tiny_model, image, tiny_size, device)
        tiny_mask = predict_crop_mask(crop_model, image, tiny_bbox, crop_size, device, args.threshold)
        save_combo_result(root, "03_picosam2_tinybox", frame_path, image, tiny_mask, tiny_bbox, records, f"{tiny_score:.6f}")

    for combo in combos:
        save_experiment_record(root / combo / "predictions.csv", records[combo])
    print(f"[Done] {root}")


def save_combo_result(
    root: Path,
    combo: str,
    frame_path: Path,
    image: Image.Image,
    mask: Image.Image,
    bbox: list[float] | None,
    records: dict[str, list[dict]],
    score: str,
) -> None:
    mask_path = root / combo / "masks" / frame_output_name(frame_path, "mask")
    overlay_path = root / combo / "frames" / frame_output_name(frame_path, "seg")
    label = combo
    overlay = overlay_mask(image, mask, bbox, label)
    mask.save(mask_path, compress_level=1)
    overlay.save(overlay_path, compress_level=1)
    records[combo].append(
        {
            "image": str(frame_path),
            "mask": str(mask_path),
            "overlay": str(overlay_path),
            "bbox": "" if bbox is None else " ".join(f"{v:.2f}" for v in bbox),
            "score": score,
        }
    )


def summarize_models(args: argparse.Namespace) -> None:
    paths = [Path(p) for p in args.checkpoints]
    summary = []
    for path in paths:
        if not path.exists():
            summary.append({"path": str(path), "missing": True})
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        params = int(ckpt.get("param_count", 0))
        summary.append(
            {
                "path": str(path),
                "file_mb": path.stat().st_size / (1024 * 1024),
                "params": params,
                "fp32_param_mb": params * 4 / (1024 * 1024),
                "int8_param_mb_estimate": params / (1024 * 1024),
                "metrics": ckpt.get("metrics", {}),
            }
        )
    print(json.dumps(summary, indent=2))


def train_all(args: argparse.Namespace) -> None:
    seg_common = {
        "epochs": args.seg_epochs,
        "num_workers": args.num_workers,
        "base_channels": args.seg_base_channels,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "amp": args.amp,
        "jitter": args.jitter,
        "crop_width": args.crop_width,
        "crop_height": args.crop_height,
        "full_width": args.full_width,
        "full_height": args.full_height,
        "run_dir": None,
    }
    train_segmentation(argparse.Namespace(mode="crop", batch_size=args.crop_batch_size, **seg_common))
    train_segmentation(argparse.Namespace(mode="full", batch_size=args.full_batch_size, **seg_common))
    train_tinybox(
        argparse.Namespace(
            epochs=args.box_epochs,
            batch_size=args.box_batch_size,
            num_workers=args.num_workers,
            base_channels=args.box_base_channels,
            width=args.box_width,
            height=args.box_height,
            lr=args.lr,
            weight_decay=args.weight_decay,
            box_loss_weight=args.box_loss_weight,
            seed=args.seed,
            run_dir=None,
        )
    )
    summarize_models(
        argparse.Namespace(
            checkpoints=[
                str(PICOSAM_ROOT / f"crop_{args.seg_base_channels}ch" / "best.pt"),
                str(PICOSAM_ROOT / f"full_{args.seg_base_channels}ch" / "best.pt"),
                str(TINYBOX_ROOT / f"tinybox_{args.box_base_channels}ch" / "best.pt"),
            ]
        )
    )
    if not args.skip_output:
        run_output_experiments(
            argparse.Namespace(
                output_id=args.output_id,
                crop_checkpoint=str(PICOSAM_ROOT / f"crop_{args.seg_base_channels}ch" / "best.pt"),
                full_checkpoint=str(PICOSAM_ROOT / f"full_{args.seg_base_channels}ch" / "best.pt"),
                tinybox_checkpoint=str(TINYBOX_ROOT / f"tinybox_{args.box_base_channels}ch" / "best.pt"),
                output_root=args.output_root,
                threshold=args.threshold,
                max_images=args.max_images,
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PicoSAM2-style SMAS segmentation and lightweight bbox experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    seg = sub.add_parser("train-seg", help="Train PicoSAM2-style U-Net segmentation.")
    seg.add_argument("--mode", choices=["crop", "full"], required=True)
    seg.add_argument("--epochs", type=int, default=8)
    seg.add_argument("--batch-size", type=int, default=16)
    seg.add_argument("--num-workers", type=int, default=0)
    seg.add_argument("--base-channels", type=int, default=32)
    seg.add_argument("--lr", type=float, default=1e-3)
    seg.add_argument("--weight-decay", type=float, default=1e-4)
    seg.add_argument("--seed", type=int, default=42)
    seg.add_argument("--no-amp", dest="amp", action="store_false")
    seg.set_defaults(amp=True)
    seg.add_argument("--jitter", type=float, default=0.08)
    seg.add_argument("--crop-width", type=int, default=256)
    seg.add_argument("--crop-height", type=int, default=128)
    seg.add_argument("--full-width", type=int, default=384)
    seg.add_argument("--full-height", type=int, default=224)
    seg.add_argument("--run-dir", default=None)

    box = sub.add_parser("train-box", help="Train tiny bbox regression detector.")
    box.add_argument("--epochs", type=int, default=20)
    box.add_argument("--batch-size", type=int, default=32)
    box.add_argument("--num-workers", type=int, default=0)
    box.add_argument("--base-channels", type=int, default=16)
    box.add_argument("--width", type=int, default=320)
    box.add_argument("--height", type=int, default=192)
    box.add_argument("--lr", type=float, default=1e-3)
    box.add_argument("--weight-decay", type=float, default=1e-4)
    box.add_argument("--box-loss-weight", type=float, default=10.0)
    box.add_argument("--seed", type=int, default=42)
    box.add_argument("--run-dir", default=None)

    infer = sub.add_parser("run-output", help="Run all three output-folder experiments.")
    infer.add_argument("output_id", help="Example: 25")
    infer.add_argument("--crop-checkpoint", default=str(PICOSAM_ROOT / "crop_32ch" / "best.pt"))
    infer.add_argument("--full-checkpoint", default=str(PICOSAM_ROOT / "full_32ch" / "best.pt"))
    infer.add_argument("--tinybox-checkpoint", default=str(TINYBOX_ROOT / "tinybox_16ch" / "best.pt"))
    infer.add_argument("--output-root", default=str(EXPERIMENT_ROOT))
    infer.add_argument("--threshold", type=float, default=0.5)
    infer.add_argument("--max-images", type=int, default=0)

    all_train = sub.add_parser("train-all", help="Train all models and run the output25 comparison.")
    all_train.add_argument("--seg-epochs", type=int, default=8)
    all_train.add_argument("--box-epochs", type=int, default=20)
    all_train.add_argument("--crop-batch-size", type=int, default=32)
    all_train.add_argument("--full-batch-size", type=int, default=12)
    all_train.add_argument("--box-batch-size", type=int, default=64)
    all_train.add_argument("--num-workers", type=int, default=0)
    all_train.add_argument("--seg-base-channels", type=int, default=32)
    all_train.add_argument("--box-base-channels", type=int, default=16)
    all_train.add_argument("--lr", type=float, default=1e-3)
    all_train.add_argument("--weight-decay", type=float, default=1e-4)
    all_train.add_argument("--box-loss-weight", type=float, default=10.0)
    all_train.add_argument("--seed", type=int, default=42)
    all_train.add_argument("--no-amp", dest="amp", action="store_false")
    all_train.set_defaults(amp=True)
    all_train.add_argument("--jitter", type=float, default=0.08)
    all_train.add_argument("--crop-width", type=int, default=256)
    all_train.add_argument("--crop-height", type=int, default=128)
    all_train.add_argument("--full-width", type=int, default=384)
    all_train.add_argument("--full-height", type=int, default=224)
    all_train.add_argument("--box-width", type=int, default=320)
    all_train.add_argument("--box-height", type=int, default=192)
    all_train.add_argument("--output-id", default="25")
    all_train.add_argument("--output-root", default=str(EXPERIMENT_ROOT))
    all_train.add_argument("--threshold", type=float, default=0.5)
    all_train.add_argument("--max-images", type=int, default=0)
    all_train.add_argument("--skip-output", action="store_true")

    summary = sub.add_parser("summary", help="Print checkpoint sizes and metrics.")
    summary.add_argument("checkpoints", nargs="+")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train-seg":
        train_segmentation(args)
    elif args.command == "train-box":
        train_tinybox(args)
    elif args.command == "run-output":
        run_output_experiments(args)
    elif args.command == "train-all":
        train_all(args)
    elif args.command == "summary":
        summarize_models(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
