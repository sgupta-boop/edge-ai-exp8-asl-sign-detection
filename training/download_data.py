"""Download the Kaggle ASL Alphabet dataset (grassknoted/asl-alphabet).

Keeps A-Z + 'nothing' (27 classes); 'del' and 'space' are ignored later by train.py.
Requires a Kaggle API token in ~/.kaggle (kaggle.json or access_token).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main():
    DATA.mkdir(exist_ok=True)
    train_dir = DATA / "asl_alphabet_train" / "asl_alphabet_train"
    if train_dir.exists():
        print("Dataset already present:", train_dir)
        return
    subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download",
         "grassknoted/asl-alphabet", "-p", str(DATA), "--unzip"],
        check=True,
    )
    print("Downloaded to", DATA)


if __name__ == "__main__":
    main()
