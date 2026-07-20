import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as VF
from tqdm import tqdm

from common import DATA_ROOT, RESAMPLE_NEAREST, Size2D, tensor_from_image
from model import PicoSAM2BaseUNet, count_params, reparameterize_model


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INIT = BASE_DIR / "checkpoints" / "picosam2_unet_320x192.pt"
DEFAULT_RUN_DIR = BASE_DIR / "runs" / "picosam2_unet_320x192"
CLASS_NAMES = ["background", "dermis", "smas", "bone"]
MASK_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff"}
SAMPLE_NAME_PATTERN = re.compile(r"^(output\d+)_frame_+(\d+)$", flags=re.IGNORECASE)


def parse_outputs(text: str) -> set[str]:
    outputs = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower().startswith("output"):
            outputs.add(f"output{int(token[6:])}")
        else:
            outputs.add(f"output{int(token)}")
    return outputs


def parse_sample_name(path: Path) -> tuple[str, int]:
    match = SAMPLE_NAME_PATTERN.match(path.stem)
    if not match:
        raise ValueError(
            f"Mask filename must look like output1_frame_00001.png, got: {path.name}"
        )
    return match.group(1).lower(), int(match.group(2))


class LayerSegmentationDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        outputs: set[str],
        image_size: Size2D,
        train: bool,
        max_output42_frame: int | None,
    ):
        image_root = data_root / "images" / split
        mask_root = data_root / "masks" / split
        if not image_root.is_dir():
            raise FileNotFoundError(f"Missing image split folder: {image_root}")
        if not mask_root.is_dir():
            raise FileNotFoundError(f"Missing mask split folder: {mask_root}")

        self.image_size = image_size
        self.train = train
        self.rows = []
        missing = []
        mask_paths = sorted(
            path
            for path in mask_root.rglob("*")
            if path.is_file() and path.suffix.lower() in MASK_EXTENSIONS
        )
        for mask_path in mask_paths:
            relative_path = mask_path.relative_to(mask_root)
            output, frame_index = parse_sample_name(mask_path)
            if output not in outputs:
                continue
            if (
                max_output42_frame is not None
                and output == "output42"
                and frame_index > max_output42_frame
            ):
                continue
            image_path = image_root / relative_path
            if image_path.exists():
                self.rows.append((image_path, mask_path, mask_path.stem))
            else:
                missing.append(str(relative_path))
        if not self.rows:
            raise RuntimeError(
                f"No paired samples for split={split}, outputs={sorted(outputs)} in {data_root}. "
                f"Missing image examples: {missing[:5]}"
            )
        if missing:
            print(f"[Warn] skipped {len(missing)} masks without a matching image.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        image_path, mask_path, sample_id = self.rows[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(mask_path) as source:
            mask = source.copy()
        if self.train and np.random.random() < 0.5:
            image = VF.hflip(image)
            mask = VF.hflip(mask)
        return tensor_from_image(image, self.image_size), make_target(mask, self.image_size), sample_id


def make_target(mask: Image.Image, size: Size2D) -> torch.Tensor:
    resized = mask.resize((size.width, size.height), RESAMPLE_NEAREST)
    target = np.asarray(resized)
    if target.ndim != 2:
        raise ValueError(f"Expected a single-channel class-index mask, got shape={target.shape}")
    invalid = np.unique(target[(target < 0) | (target >= len(CLASS_NAMES))])
    if invalid.size:
        raise ValueError(
            f"Mask contains invalid class values {invalid.tolist()}; expected integers 0-{len(CLASS_NAMES) - 1}."
        )
    return torch.from_numpy(target.astype(np.int64, copy=True))


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_backbone_init(model: torch.nn.Module, checkpoint: Path) -> int:
    if not checkpoint.exists():
        return 0
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = ckpt.get("model", ckpt)
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    target.update(compatible)
    model.load_state_dict(target)
    return len(compatible)


def multiclass_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    probs = probs[:, 1:]
    one_hot = one_hot[:, 1:]
    dims = (0, 2, 3)
    inter = torch.sum(probs * one_hot, dim=dims)
    denom = torch.sum(probs + one_hot, dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def seg_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets, weight=weights) + multiclass_dice_loss(logits, targets)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    scores = {name: {"dice": [], "iou": []} for name in CLASS_NAMES[1:]}
    for images, targets, _ in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        preds = torch.argmax(model(images), dim=1)
        for class_id, name in enumerate(CLASS_NAMES[1:], start=1):
            pred = preds == class_id
            target = targets == class_id
            dims = tuple(range(1, pred.ndim))
            inter = torch.sum(pred & target, dim=dims).float()
            pred_sum = torch.sum(pred, dim=dims).float()
            target_sum = torch.sum(target, dim=dims).float()
            union = pred_sum + target_sum - inter
            valid = target_sum > 0
            if valid.any():
                dice = (2 * inter + 1e-6) / (pred_sum + target_sum + 1e-6)
                iou = (inter + 1e-6) / (union + 1e-6)
                scores[name]["dice"].extend(dice[valid].detach().cpu().tolist())
                scores[name]["iou"].extend(iou[valid].detach().cpu().tolist())

    metrics = {"images": len(loader.dataset)}
    ious = []
    dices = []
    for name, values in scores.items():
        dice = float(np.mean(values["dice"])) if values["dice"] else 0.0
        iou = float(np.mean(values["iou"])) if values["iou"] else 0.0
        metrics[f"{name}_dice"] = dice
        metrics[f"{name}_iou"] = iou
        dices.append(dice)
        ious.append(iou)
    metrics["mean_dice"] = float(np.mean(dices)) if dices else 0.0
    metrics["mean_iou"] = float(np.mean(ious)) if ious else 0.0
    return metrics


def save_checkpoint(path: Path, model: torch.nn.Module, metrics: dict, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_args = vars(args).copy()
    payload_args["num_classes"] = len(CLASS_NAMES)
    payload_args["class_names"] = CLASS_NAMES
    torch.save(
        {
            "model": model.state_dict(),
            "metrics": metrics,
            "args": payload_args,
            "param_count": count_params(model),
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = Size2D(args.width, args.height)
    data_root = Path(args.data_root).resolve()
    train_outputs = parse_outputs(args.train_outputs)
    val_outputs = parse_outputs(args.val_outputs)
    run_dir = Path(args.run_dir) if args.run_dir else DEFAULT_RUN_DIR

    train_ds = LayerSegmentationDataset(
        data_root,
        args.train_split,
        train_outputs,
        image_size,
        train=True,
        max_output42_frame=args.max_output42_frame,
    )
    val_ds = LayerSegmentationDataset(
        data_root,
        args.val_split,
        val_outputs,
        image_size,
        train=False,
        max_output42_frame=None,
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

    model = PicoSAM2BaseUNet(base_channels=args.base_channels, out_channels=len(CLASS_NAMES)).to(device)
    init_count = 0 if args.no_init else load_backbone_init(model, Path(args.init_from))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    weights = torch.tensor(
        [args.background_weight, args.dermis_weight, args.smas_weight, args.bone_weight],
        dtype=torch.float32,
        device=device,
    )

    best_iou = -1.0
    log_rows = []
    print(f"[Train] classes={CLASS_NAMES}")
    print(f"[Train] data_root={data_root}")
    print(f"[Train] train_outputs={sorted(train_outputs)} val_outputs={sorted(val_outputs)}")
    print(f"[Train] train={len(train_ds)} val={len(val_ds)} max_output42_frame={args.max_output42_frame}")
    print(f"[Train] params={count_params(model):,} init_keys={init_count} device={device} run_dir={run_dir}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        losses = []
        for images, targets, _ in tqdm(train_loader, desc=f"train-{epoch}", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(images)
                loss = seg_loss(logits, targets, weights)
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
            f"mean_iou={metrics['mean_iou']:.4f} dermis_iou={metrics['dermis_iou']:.4f} "
            f"smas_iou={metrics['smas_iou']:.4f} bone_iou={metrics['bone_iou']:.4f} {metrics['seconds']:.1f}s"
        )
        save_checkpoint(run_dir / "last.pt", model, metrics, args)
        if metrics["mean_iou"] > best_iou:
            best_iou = metrics["mean_iou"]
            save_checkpoint(run_dir / "best.pt", model, metrics, args)

    # Export a reparameterized (single-branch) checkpoint of the best model for
    # fast inference. RepVGG fuses its multi-branch train graph into plain 3x3 convs.
    best_path = run_dir / "best.pt"
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        deploy_model = PicoSAM2BaseUNet(
            base_channels=args.base_channels, out_channels=len(CLASS_NAMES)
        )
        deploy_model.load_state_dict(best_ckpt["model"])
        deploy_model = reparameterize_model(deploy_model, inplace=True)
        save_checkpoint(run_dir / "best_deploy.pt", deploy_model, best_ckpt.get("metrics", {}), args)
        print(f"[Deploy] reparameterized best model -> {run_dir / 'best_deploy.pt'}")

    summary = {
        "model": "PicoSAM2BaseUNet",
        "training": "full-frame image-only mutually exclusive dermis+SMAS+bone multiclass segmentation",
        "classes": CLASS_NAMES,
        "train_outputs": sorted(train_outputs),
        "val_outputs": sorted(val_outputs),
        "max_output42_frame": args.max_output42_frame,
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
        description="Train mutually exclusive dermis + SMAS + bone multiclass segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--train-outputs", default="1,20,30,42")
    parser.add_argument("--val-outputs", default="10")
    parser.add_argument("--max-output42-frame", type=int, default=290)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--background-weight", type=float, default=0.15)
    parser.add_argument("--dermis-weight", type=float, default=1.0)
    parser.add_argument("--smas-weight", type=float, default=1.4)
    parser.add_argument("--bone-weight", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--init-from", default=str(DEFAULT_INIT))
    parser.add_argument("--no-init", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
