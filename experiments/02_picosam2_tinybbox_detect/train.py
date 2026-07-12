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
        description="Train only the PicoSAM2 crop U-Net + TinyBox detector experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seg-epochs", type=int, default=8)
    parser.add_argument("--box-epochs", type=int, default=20)
    parser.add_argument("--seg-batch-size", type=int, default=32)
    parser.add_argument("--box-batch-size", type=int, default=64)
    parser.add_argument("--seg-base-channels", type=int, default=32)
    parser.add_argument("--box-base-channels", type=int, default=16)
    parser.add_argument("--skip-seg", action="store_true")
    parser.add_argument("--skip-box", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.skip_seg:
        seg_cmd = [
            PYTHON_EXE,
            TOOL,
            "train-seg",
            "--mode",
            "crop",
            "--epochs",
            args.seg_epochs,
            "--batch-size",
            args.seg_batch_size,
            "--base-channels",
            args.seg_base_channels,
            "--run-dir",
            EXP_DIR / "runs_picosam2" / f"crop_{args.seg_base_channels}ch",
        ]
        if args.no_amp:
            seg_cmd.append("--no-amp")
        run(seg_cmd)
    if not args.skip_box:
        run(
            [
                PYTHON_EXE,
                TOOL,
                "train-box",
                "--epochs",
                args.box_epochs,
                "--batch-size",
                args.box_batch_size,
                "--base-channels",
                args.box_base_channels,
                "--run-dir",
                EXP_DIR / "runs_tinybox" / f"tinybox_{args.box_base_channels}ch",
            ]
        )


if __name__ == "__main__":
    main()
