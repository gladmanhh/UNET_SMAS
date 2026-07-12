import argparse
import subprocess
import sys
from argparse import Namespace
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_VENV_PYTHON = PROJECT_ROOT.parent / "CLASSYS-BEAUTY" / ".venv" / "Scripts" / "python.exe"
PYTHON_EXE = DEFAULT_VENV_PYTHON if DEFAULT_VENV_PYTHON.exists() else Path(sys.executable)


def format_command(cmd: list[object]) -> str:
    parts = []
    for arg in cmd:
        text = str(arg)
        if " " in text:
            text = f'"{text}"'
        parts.append(text)
    return " ".join(parts)


def run(cmd: list[object]) -> None:
    cmd = [str(arg) for arg in cmd]
    print(f"\n[Run] {format_command(cmd)}\n", flush=True)
    subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)


def resolve_run_dir(run_name: str) -> Path:
    path = Path(run_name)
    if path.is_absolute():
        return path
    if len(path.parts) > 1:
        return (SCRIPT_DIR / path).resolve()
    return SCRIPT_DIR / "runs" / run_name


def latest_checkpoint() -> Path:
    candidates = sorted(
        (SCRIPT_DIR / "runs").glob("*/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No best.pt checkpoint found under: {SCRIPT_DIR / 'runs'}")
    return candidates[0]


def resolve_checkpoint(value: str | None) -> Path:
    if value is None or value.lower() == "latest":
        return latest_checkpoint()
    path = Path(value)
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def add_prepare_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--overwrite", action="store_true", help="Rebuild linked/copied dataset files.")
    parser.add_argument("--bbox-pad", type=int, default=0, help="Padding in pixels around mask bbox.")


def run_prepare(args: Namespace) -> None:
    cmd: list[object] = [PYTHON_EXE, SCRIPT_DIR / "prepare_bbox_dataset.py"]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.bbox_pad != 0:
        cmd.extend(["--bbox-pad", args.bbox_pad])
    run(cmd)


def run_train(args: Namespace) -> None:
    if not args.skip_prepare:
        run_prepare(args)

    cmd: list[object] = [
        PYTHON_EXE,
        SCRIPT_DIR / "train_detector.py",
        "--epochs",
        args.epochs,
        "--batch-size",
        args.batch_size,
        "--lr",
        args.lr,
    ]
    if args.run_name:
        cmd.extend(["--run-dir", resolve_run_dir(args.run_name)])
    if args.amp:
        cmd.append("--amp")
    run(cmd)


def run_infer(args: Namespace) -> None:
    output_id = args.output_id
    if not output_id:
        output_id = input("Output number (1 -> output1) [1]: ").strip() or "1"

    checkpoint = resolve_checkpoint(args.checkpoint)
    cmd: list[object] = [
        PYTHON_EXE,
        SCRIPT_DIR / "infer_detector.py",
        output_id,
        "--checkpoint",
        checkpoint,
        "--score-thresh",
        args.score_thresh,
    ]
    if args.max_images > 0:
        cmd.extend(["--max-images", args.max_images])
    if args.skip_existing:
        cmd.append("--skip-existing")
    run(cmd)


def ask_text(prompt: str, default: str) -> str:
    answer = input(f"{prompt} [{default}]: ").strip()
    return answer or default


def ask_int(prompt: str, default: int) -> int:
    return int(ask_text(prompt, str(default)))


def interactive_args() -> Namespace:
    print("CNN_SMAS launcher")
    print("1. infer: run detection on output frames")
    print("2. train: prepare data, then train")
    print("3. prepare: prepare data only")
    choice = ask_text("Select", "1")

    if choice == "2":
        return Namespace(
            command="train",
            epochs=ask_int("Epochs", 12),
            batch_size=ask_int("Batch size", 4),
            lr=float(ask_text("Learning rate", "0.0001")),
            run_name=ask_text("Run name (blank = timestamp)", ""),
            amp=ask_text("AMP y/n", "n").lower().startswith("y"),
            skip_prepare=False,
            overwrite=False,
            bbox_pad=0,
        )
    if choice == "3":
        return Namespace(command="prepare", overwrite=False, bbox_pad=0)

    return Namespace(
        command="infer",
        output_id=ask_text("Output number", "1"),
        checkpoint="latest",
        score_thresh=float(ask_text("Score threshold", "0.5")),
        max_images=0,
        skip_existing=False,
    )


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description="Small launcher for CNN_SMAS dataset prep, training, and inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    prepare = subparsers.add_parser("prepare", help="Prepare bbox dataset from CLASSYS-BEAUTY.")
    add_prepare_options(prepare)

    train = subparsers.add_parser("train", help="Prepare data, then train the detector.")
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--run-name", default=None, help="Folder name under runs/. Blank uses timestamp.")
    train.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    train.add_argument("--skip-prepare", action="store_true", help="Skip prepare_bbox_dataset.py.")
    add_prepare_options(train)

    infer = subparsers.add_parser("infer", help="Run inference on an output frame folder.")
    infer.add_argument("output_id", nargs="?", default=None, help="Example: 1 resolves to output1.")
    infer.add_argument("--checkpoint", default="latest", help="Path to best.pt, or latest.")
    infer.add_argument("--score-thresh", type=float, default=0.5)
    infer.add_argument("--max-images", type=int, default=0, help="0 means all images.")
    infer.add_argument("--skip-existing", action="store_true")

    args = parser.parse_args()
    if args.command is None:
        return interactive_args()
    return args


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        run_prepare(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "infer":
        run_infer(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
