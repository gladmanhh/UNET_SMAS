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

from common import DATA_ROOT, PROJECT_ROOT, RESAMPLE_NEAREST, Size2D, tensor_from_image
from model import PicoSAM2BaseUNet, count_params


BASE_DIR = Path(__file__).resolve().parent
LOCAL_DERMIS_ROOT = PROJECT_ROOT.parent / "beauty" / "analog" / "analog_result_24_v1"
LOCAL_BONE_ROOT = PROJECT_ROOT.parent / "CLASSYS-BEAUTY" / "tools" / "outputs"
DEFAULT_DERMIS_ROOT = LOCAL_DERMIS_ROOT if LOCAL_DERMIS_ROOT.exists() else DATA_ROOT / "dermis"
DEFAULT_BONE_ROOT = LOCAL_BONE_ROOT if LOCAL_BONE_ROOT.exists() else DATA_ROOT / "bone"
DEFAULT_INIT = BASE_DIR / "runs_multiclass" / "dermis_smas_32ch" / "best.pt"
DEFAULT_RUN_DIR = BASE_DIR / "runs_bone_multiclass" / "dermis_smas_bone_32ch"
CLASS_NAMES = ["background", "dermis", "smas", "bone"]


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


def frame_number(name: str) -> int:
    match = re.search(r"(\d+)", name)
    if not match:
        return -1
    return int(match.group(1))


class DermisSMASBoneDataset(Dataset):
    def __init__(
        self,
        annotations: list[dict],
        outputs: set[str],
        image_size: Size2D,
        dermis_root: Path,
        bone_root: Path,
        train: bool,
        max_output42_frame: int | None,
    ):
        rows = [row for row in annotations if row["output"] in outputs]
        if max_output42_frame is not None:
            rows = [
                row
                for row in rows
                if row["output"] != "output42" or frame_number(row["frame"]) <= max_output42_frame
            ]
        self.image_size = image_size
        self.dermis_root = dermis_root
        self.bone_root = bone_root
        self.train = train
        self.rows = []
        missing = []
        for row in rows:
            paths = [PROJECT_ROOT / row["image_path"], PROJECT_ROOT / row["mask_path"], self.dermis_path(row), self.bone_path(row)]
            if all(path.exists() for path in paths):
                self.rows.append(row)
            else:
                missing.append(row["id"])
        if not self.rows:
            raise RuntimeError(f"No usable rows. Missing examples: {missing[:5]}")
        if missing:
            print(f"[Warn] skipped {len(missing)} rows without matching image/dermis/SMAS/bone masks.")

    def dermis_path(self, row: dict) -> Path:
        return self.dermis_root / row["output"] / "dermis_masks" / row["frame"]

    def bone_path(self, row: dict) -> Path:
        return self.bone_root / row["output"] / "labels" / row["frame"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(PROJECT_ROOT / row["image_path"]).convert("RGB")
        dermis = Image.open(self.dermis_path(row)).convert("L")
        smas = Image.open(PROJECT_ROOT / row["mask_path"]).convert("L")
        bone = Image.open(self.bone_path(row)).convert("L")
        if self.train and np.random.random() < 0.5:
            image = VF.hflip(image)
            dermis = VF.hflip(dermis)
            smas = VF.hflip(smas)
            bone = VF.hflip(bone)
        return tensor_from_image(image, self.image_size), make_target(dermis, smas, bone, self.image_size), row["id"]


def make_target(dermis: Image.Image, smas: Image.Image, bone: Image.Image, size: Size2D) -> torch.Tensor:
    dermis_arr = np.array(dermis.resize((size.width, size.height), RESAMPLE_NEAREST).convert("L")) > 0
    smas_arr = np.array(smas.resize((size.width, size.height), RESAMPLE_NEAREST).convert("L")) > 0
    bone_arr = np.array(bone.resize((size.width, size.height), RESAMPLE_NEAREST).convert("L")) > 0
    target = np.zeros((size.height, size.width), dtype=np.int64)
    target[dermis_arr] = 1
    target[smas_arr] = 2
    target[bone_arr] = 3
    return torch.from_numpy(target)


def load_annotations(path: Path = DATA_ROOT / "annotations.json") -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["annotations"]


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
    for images, targets, _ in tqdm(loader, desc="eval-bone-multi", leave=False):
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
    annotations = load_annotations(Path(args.annotations))
    train_outputs = parse_outputs(args.train_outputs)
    val_outputs = parse_outputs(args.val_outputs)
    run_dir = Path(args.run_dir) if args.run_dir else DEFAULT_RUN_DIR
    dermis_root = Path(args.dermis_root)
    bone_root = Path(args.bone_root)

    train_ds = DermisSMASBoneDataset(
        annotations,
        train_outputs,
        image_size,
        dermis_root,
        bone_root,
        train=True,
        max_output42_frame=args.max_output42_frame,
    )
    val_ds = DermisSMASBoneDataset(
        annotations,
        val_outputs,
        image_size,
        dermis_root,
        bone_root,
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
    print(f"[TrainBoneMulti] classes={CLASS_NAMES}")
    print(f"[TrainBoneMulti] train_outputs={sorted(train_outputs)} val_outputs={sorted(val_outputs)}")
    print(f"[TrainBoneMulti] train={len(train_ds)} val={len(val_ds)} max_output42_frame={args.max_output42_frame}")
    print(f"[TrainBoneMulti] params={count_params(model):,} init_keys={init_count} device={device} run_dir={run_dir}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        losses = []
        for images, targets, _ in tqdm(train_loader, desc=f"train-bone-multi-{epoch}", leave=False):
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
    parser.add_argument("--annotations", default=str(DATA_ROOT / "annotations.json"))
    parser.add_argument("--dermis-root", default=str(DEFAULT_DERMIS_ROOT))
    parser.add_argument("--bone-root", default=str(DEFAULT_BONE_ROOT))
    parser.add_argument("--train-outputs", default="1,20,30,42")
    parser.add_argument("--val-outputs", default="10")
    parser.add_argument("--max-output42-frame", type=int, default=290)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=224)
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
