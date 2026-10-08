# Expt 8 – Edge AI: ASL alphabet recognition on an Android edge device

| Path | What |
|---|---|
| `training/` | `download_data.py` → `train.py` → `quantize.py` (Kaggle ASL Alphabet, 27 classes) |
| `training/out/` | Curves, confusion matrix, reports, `model_data.h`. The models and `model_data.cc` come from Hugging Face (see below) |
| `android/SignEdge/` | Kotlin + CameraX + TFLite app: live inference, torch/vibration actuation, latency/memory overlay |
| `report/results.md` | Results tables, observations, post-lab and viva answers |

## Large files (not in git)
The models and the C array are hosted on Hugging Face: https://huggingface.co/satvik4577/asl-alphabet-tinycnn-int8. Fetch them with:
```
pip install huggingface_hub
python training/fetch_models.py
```
This command puts them in `training/out/` and copies `asl_int8.tflite` into the app's `assets/`. **The Android app needs this step before it can build and run.**

The ~1 GB dataset is not hosted. `training/download_data.py` re-downloads it from Kaggle (you need a Kaggle API token).

## Re-train (optional; replaces the downloaded models)
```
pip install -r training/requirements.txt
python training/download_data.py
python training/train.py
python training/quantize.py
```
`quantize.py` copies the model and labels into the app's `assets/`.

## Run on a phone
1. Install Android Studio, then **File → Open** `android/SignEdge`. Let Gradle sync; accept any suggested AGP or Gradle upgrade.
2. On the phone, enable Developer options and USB debugging, then connect it over USB.
3. Click **Run ▶**. Grant camera permission.
4. Hold a sign inside the white box, against a plain background in good light. When a letter holds ≥ 80% confidence for 5 frames, the torch flashes, the phone vibrates, and the letter is appended.
5. To read profiling numbers, run `adb logcat -s EdgeProfile`, or use the on-screen overlay. Tap **Reset stats** before a clean measurement.
