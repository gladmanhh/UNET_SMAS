import argparse
import subprocess
import sys
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[1]
TOOL = PROJECT_ROOT / "experiments" / "_combo_tools" / "picosam2_smas.py"
DEFAULT_VENV_PYTHON = PROJECT_ROOT.parent / "CLASSYS-BEAUTY" / ".venv" / "Scripts" / "python.exe"
PYTHON_EXE = DEFAULT_VENV_PYTHON if DEFAULT_VENV_PYTHON.exists() else Path(sys.executable)


def run(cmd: list[object]) -> None:
    cmd = [str(part) for part in cmd]
    print("[Run] " + " ".join(f'"{x}"' if " " in x else x for x in cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train only the crop PicoSAM2-style U-Net for the Faster R-CNN + PicoSAM2 experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--crop-width", type=int, default=256)
    parser.add_argument("--crop-height", type=int, default=128)
    parser.add_argument("--run-dir", default=str(EXP_DIR / "runs_picosam2" / "crop_32ch"))
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cmd = [
        PYTHON_EXE,
        TOOL,
        "train-seg",
        "--mode",
        "crop",
        "--epochs",
        args.epochs,
        "--batch-size",
        args.batch_size,
        "--base-channels",
        args.base_channels,
        "--crop-width",
        args.crop_width,
        "--crop-height",
        args.crop_height,
        "--run-dir",
        args.run_dir,
    ]
    if args.no_amp:
        cmd.append("--no-amp")
    run(cmd)


if __name__ == "__main__":
    main()
