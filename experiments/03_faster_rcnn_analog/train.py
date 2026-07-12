import subprocess
import sys
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[1]
DEFAULT_VENV_PYTHON = PROJECT_ROOT.parent / "CLASSYS-BEAUTY" / ".venv" / "Scripts" / "python.exe"
PYTHON_EXE = DEFAULT_VENV_PYTHON if DEFAULT_VENV_PYTHON.exists() else Path(sys.executable)


def run(cmd: list[object]) -> None:
    cmd = [str(part) for part in cmd]
    print("[Run] " + " ".join(f'"{x}"' if " " in x else x for x in cmd), flush=True)
    subprocess.run(cmd, cwd=EXP_DIR, check=True)


def main() -> None:
    run([PYTHON_EXE, EXP_DIR / "prepare_bbox_dataset.py"])
    run([PYTHON_EXE, EXP_DIR / "train_detector.py"])


if __name__ == "__main__":
    main()
