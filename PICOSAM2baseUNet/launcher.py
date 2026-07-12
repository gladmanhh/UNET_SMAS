import argparse
import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_CHECKPOINT = BASE_DIR / "checkpoints" / "picosam2_unet_320x192.pt"
LOCAL_VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
SHARED_VENV_PYTHON = PROJECT_ROOT.parent / "CLASSYS-BEAUTY" / ".venv" / "Scripts" / "python.exe"
PYTHON_EXE = next(
    (path for path in (LOCAL_VENV_PYTHON, SHARED_VENV_PYTHON) if path.exists()),
    Path(sys.executable),
)


def run(cmd: list[object]) -> None:
    cmd = [str(part) for part in cmd]
    print("[Run] " + " ".join(f'"{x}"' if " " in x else x for x in cmd), flush=True)
    subprocess.run(cmd, cwd=BASE_DIR, check=True)


def ask(prompt: str, default: str) -> str:
    answer = input(f"{prompt} [{default}]: ").strip()
    return answer or default


def interactive() -> argparse.Namespace:
    print("PICOSAM2baseUNet launcher")
    print("1. infer SMAS-only output frames")
    print("2. train SMAS-only model")
    print("3. infer dermis+SMAS output frames")
    print("4. train dermis+SMAS model")
    print("5. infer 5-layer output video")
    print("6. train dermis+SMAS+bone model")
    print("7. checkpoint summary")
    choice = ask("Select", "1")
    if choice == "2":
        return argparse.Namespace(
            command="train",
            output_id=None,
            epochs=ask("Epochs", "8"),
            batch_size=ask("Batch size", "12"),
            threshold="0.5",
        )
    if choice == "3":
        return argparse.Namespace(
            command="infer-multiclass",
            output_id=ask("Output number", "25"),
            epochs=None,
            batch_size=None,
            threshold=None,
        )
    if choice == "4":
        return argparse.Namespace(
            command="train-multiclass",
            output_id=None,
            epochs=ask("Epochs", "8"),
            batch_size=ask("Batch size", "12"),
            threshold=None,
        )
    if choice == "5":
        return argparse.Namespace(
            command="infer-bone",
            output_id=ask("Output number", "25"),
            epochs=None,
            batch_size=None,
            threshold=None,
            inference_size=None,
            pipeline_depth=3,
        )
    if choice == "6":
        return argparse.Namespace(
            command="train-bone",
            output_id=None,
            epochs=ask("Epochs", "8"),
            batch_size=ask("Batch size", "12"),
            threshold=None,
        )
    if choice == "7":
        return argparse.Namespace(command="summary", output_id=None, epochs=None, batch_size=None, threshold=None)
    return argparse.Namespace(
        command="infer",
        output_id=ask("Output number", "25"),
        epochs=None,
        batch_size=None,
        threshold=ask("Threshold", "0.5"),
    )


def parse_args() -> argparse.Namespace:
    raw_args = sys.argv[1:]
    if raw_args and (raw_args[0].isdigit() or raw_args[0].lower().startswith("output")):
        shortcut = argparse.ArgumentParser(
            description="Shortcut inference options.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        shortcut.add_argument("output_id")
        shortcut.add_argument("--threshold", default="0.5")
        args = shortcut.parse_args(raw_args)
        return argparse.Namespace(
            command="infer",
            output_id=args.output_id,
            threshold=args.threshold,
        )

    parser = argparse.ArgumentParser(
        description="Launcher for image-only PicoSAM2BaseUNet training and inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")
    infer = sub.add_parser("infer", help="Run inference on CLASSYS-BEAUTY output frames.")
    infer.add_argument("output_id", nargs="?", default=None)
    infer.add_argument("--threshold", default="0.5")

    train = sub.add_parser("train", help="Train the full-frame segmentation model.")
    train.add_argument("--epochs", default="8")
    train.add_argument("--batch-size", default="12")

    infer_multi = sub.add_parser("infer-multiclass", help="Run dermis+SMAS inference on output frames.")
    infer_multi.add_argument("output_id", nargs="?", default=None)

    train_multi = sub.add_parser("train-multiclass", help="Train dermis+SMAS multiclass segmentation.")
    train_multi.add_argument("--epochs", default="8")
    train_multi.add_argument("--batch-size", default="12")

    infer_bone = sub.add_parser("infer-bone", help="Run 5-layer segmentation video inference on output frames.")
    infer_bone.add_argument("output_id", nargs="?", default=None)
    infer_bone.add_argument("--inference-size", default=None, metavar="WIDTHxHEIGHT")
    infer_bone.add_argument("--pipeline-depth", type=int, default=3)

    train_bone = sub.add_parser("train-bone", help="Train dermis+SMAS+bone multiclass segmentation.")
    train_bone.add_argument("--epochs", default="8")
    train_bone.add_argument("--batch-size", default="12")

    sub.add_parser("summary", help="Print checkpoint size and metrics.")
    args = parser.parse_args()
    if args.command is None:
        return interactive()
    if args.command == "infer" and args.output_id is None:
        args.output_id = ask("Output number", "25")
    return args


def checkpoint_summary() -> None:
    if not DEFAULT_CHECKPOINT.exists():
        print(f"Missing checkpoint: {DEFAULT_CHECKPOINT}")
        return
    import torch

    ckpt = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
    payload = {
        "checkpoint": str(DEFAULT_CHECKPOINT),
        "file_mb": DEFAULT_CHECKPOINT.stat().st_size / (1024 * 1024),
        "param_count": ckpt.get("param_count"),
        "metrics": ckpt.get("metrics", {}),
    }
    print(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "infer":
        run(
            [
                PYTHON_EXE,
                BASE_DIR / "infer.py",
                args.output_id,
                "--threshold",
                args.threshold,
            ]
        )
    elif args.command == "train":
        run([PYTHON_EXE, BASE_DIR / "train.py", "--epochs", args.epochs, "--batch-size", args.batch_size])
    elif args.command == "infer-multiclass":
        if args.output_id is None:
            args.output_id = ask("Output number", "25")
        run([PYTHON_EXE, BASE_DIR / "infer_multiclass.py", args.output_id])
    elif args.command == "train-multiclass":
        run([PYTHON_EXE, BASE_DIR / "train_multiclass.py", "--epochs", args.epochs, "--batch-size", args.batch_size])
    elif args.command == "infer-bone":
        if args.output_id is None:
            args.output_id = ask("Output number", "25")
        cmd = [PYTHON_EXE, BASE_DIR / "infer_bone_multiclass.py", args.output_id]
        if args.inference_size:
            cmd.extend(["--inference-size", args.inference_size])
        cmd.extend(["--pipeline-depth", args.pipeline_depth])
        run(cmd)
    elif args.command == "train-bone":
        run([PYTHON_EXE, BASE_DIR / "train_bone_multiclass.py", "--epochs", args.epochs, "--batch-size", args.batch_size])
    elif args.command == "summary":
        checkpoint_summary()
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
