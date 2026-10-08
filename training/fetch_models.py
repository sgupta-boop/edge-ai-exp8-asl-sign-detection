"""Download the trained models (kept out of git) from the Hugging Face Hub.

Places them in training/out/ and copies the int8 model into the Android app's assets/.
"""
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "satvik4577/asl-alphabet-tinycnn-int8"
FILES = ["asl_int8.tflite", "asl_fp32.tflite", "asl_fp32.keras", "model_data.cc"]

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ASSETS = HERE.parent / "android" / "SignEdge" / "app" / "src" / "main" / "assets"


def main():
    OUT.mkdir(exist_ok=True)
    for name in FILES:
        path = hf_hub_download(REPO, name)
        shutil.copy(path, OUT / name)
        print("downloaded", OUT / name)
    ASSETS.mkdir(parents=True, exist_ok=True)
    shutil.copy(OUT / "asl_int8.tflite", ASSETS / "asl_int8.tflite")
    print("copied asl_int8.tflite ->", ASSETS)


if __name__ == "__main__":
    main()
