# Expt 8 – Edge AI: ASL alphabet recognition on an Android edge device

This project is a tiny CNN, about 92 KB once quantized to int8. It recognises American Sign Language letters A–Z, plus "nothing", from the phone camera. Everything runs **on the phone itself**, with no cloud. When it is confident about a letter, the phone flashes its torch and vibrates.

| | |
|---|---|
| **Code** (this repo) | https://github.com/sgupta-boop/edge-ai-exp8-asl-sign-detection |
| **Model files** (Hugging Face) | https://huggingface.co/satvik4577/asl-alphabet-tinycnn-int8 |
| **Test accuracy** | 95.78% (int8), 95.69% (fp32) |

## What's in the repo

| Path | What |
|---|---|
| `training/` | `download_data.py` → `train.py` → `quantize.py` (Kaggle ASL Alphabet, 27 classes); `fetch_models.py` downloads the trained models |
| `training/out/` | Training curves, confusion matrix, reports, `model_data.h`. The models and `model_data.cc` come from Hugging Face (step 3). |
| `android/SignEdge/` | Kotlin + CameraX + TFLite app: live inference, torch/vibration actuation, latency/memory overlay |
| `report/results.md` | Results tables, observations, post-lab and viva answers |

The model files are **not stored in git**, to keep the repo small. They live on Hugging Face, and step 3 below downloads them. The ~1 GB training dataset isn't hosted anywhere by us; it comes from Kaggle and is only needed for retraining (step 6).

---

## Setup for new users

You do **not** need a GitHub or Hugging Face account. Both are public.

### 0. Install these first (one time)
- **Git:** https://git-scm.com/downloads
- **Python 3.10–3.12:** https://www.python.org/downloads/. On Windows, tick **"Add Python to PATH"** during install.
- **Only to run the phone app:**
  - Android Studio: https://developer.android.com/studio
  - an Android phone (Android 7.0 or newer) and a USB cable

Open a terminal (PowerShell on Windows, Terminal on Mac/Linux) to run the commands below.

### 1. Download the code
```bash
git clone https://github.com/sgupta-boop/edge-ai-exp8-asl-sign-detection.git
```
```bash
cd edge-ai-exp8-asl-sign-detection
```

### 2. Install the Hugging Face downloader
```bash
pip install -U huggingface_hub
```
If `pip` is not found, use `python -m pip install -U huggingface_hub`.

### 3. Download the model files
**Option A (easiest):** this script downloads every model file and also places the model inside the Android app.
```bash
python training/fetch_models.py
```

**Option B:** use the `hf` command-line tool.
```bash
hf download satvik4577/asl-alphabet-tinycnn-int8 --local-dir training/out
```
```bash
hf download satvik4577/asl-alphabet-tinycnn-int8 asl_int8.tflite --local-dir android/SignEdge/app/src/main/assets
```

**Option C (no commands):** download the files by hand.
1. Open the [Hugging Face page](https://huggingface.co/satvik4577/asl-alphabet-tinycnn-int8) and go to **Files and versions**.
2. Download each file.
3. Put `asl_int8.tflite` in `android/SignEdge/app/src/main/assets/`, and everything else in `training/out/`.

| File | Size | Used for |
|---|---|---|
| `asl_int8.tflite` | 92 KB | **The model the Android app runs** |
| `asl_fp32.tflite` | 327 KB | Float TFLite model, for comparison |
| `asl_fp32.keras` | 1 MB | Original trained Keras model |
| `model_data.cc` / `model_data.h` | 0.6 MB | int8 model as a C byte array, for microcontrollers (TFLite Micro) |

### 4. Check that the model is in place
Windows:
```bash
dir android\SignEdge\app\src\main\assets
```
Mac/Linux:
```bash
ls android/SignEdge/app/src/main/assets
```
You should see `asl_int8.tflite` (about 92 KB) and `labels.txt`. **The app will crash on start without `asl_int8.tflite`.**

### 5. Run the app on your phone
1. Open Android Studio and choose **File → Open**. Select the **`android/SignEdge`** folder, not the repo root.
2. Wait for the Gradle sync to finish (this takes a few minutes the first time). If it offers to upgrade Gradle or the Android Gradle Plugin, accept.
3. On the phone, open **Settings → About phone** and tap **Build number** 7 times. This turns on Developer options.
4. Open **Settings → Developer options** (sometimes under *System*) and turn on **USB debugging**.
5. Connect the phone by USB. Tap **Allow** on the "Allow USB debugging?" prompt.
6. In Android Studio, pick your phone from the device dropdown and press the green **Run ▶** button.
7. When the app opens, allow camera access.

**Using the app:**
- Hold one hand sign **inside the white box**. Use a **plain background** and **good light**.
- When a letter stays at ≥ 80% confidence for 5 frames, the torch flashes, the phone vibrates, the box turns green, and the letter is added to the word at the bottom.
- To type the same letter twice, drop your hand for a moment between signs.
- Buttons:
  - **Flip cam** switches between the front and back camera. Only the back camera has a torch.
  - **⌫** deletes the last letter, and **Clear** clears the word.
  - **Reset stats** restarts the latency measurements.
- The top-left overlay shows model size, memory use, inference latency (avg / p95) and FPS.

To read the profiling numbers on a computer:
```bash
adb logcat -s EdgeProfile
```
`adb` comes with Android Studio, in `Android/Sdk/platform-tools`.

### 6. Optional: retrain the model yourself
You need a free Kaggle account to download the dataset:
1. On kaggle.com, go to **Settings → API → Create New Token**.
2. Save the downloaded file to `C:\Users\<you>\.kaggle\` on Windows, or `~/.kaggle/` on Mac/Linux.

Then run, one after another:
```bash
pip install -r training/requirements.txt
```
```bash
python training/download_data.py
```
```bash
python training/train.py
```
```bash
python training/quantize.py
```
- The download is about 1 GB.
- Training takes about 25 minutes on a laptop CPU.
- `quantize.py` writes new models to `training/out/` and copies the int8 model into the app automatically.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` or `pip` not found | Reinstall Python with "Add to PATH" ticked, or try `py` / `python3` |
| `hf` not found | Use `python training/fetch_models.py` instead, or run `python -m pip install -U huggingface_hub` again |
| App crashes immediately | `asl_int8.tflite` is missing from `android/SignEdge/app/src/main/assets/`. Redo step 3. |
| Phone not shown in Android Studio | Check that USB debugging is on, try another cable, and set the USB mode to "File transfer" |
| Predictions are wrong or jumpy | Use a plain wall behind your hand and better light, and keep your hand fully inside the box. The model was trained on one person against one background, so live accuracy is lower than the 95.8% test score. |

## Contributing changes
- **Code:** ask the owner to add you under GitHub **Settings → Collaborators**. Then run `git add .`, `git commit -m "..."` and `git push`.
- **Models:** sign in with your own Hugging Face account (`hf auth login`), then open a pull request on the model repo:
  ```bash
  hf upload satvik4577/asl-alphabet-tinycnn-int8 training/out/asl_int8.tflite asl_int8.tflite --create-pr
  ```
  The owner approves it under the **Community** tab on Hugging Face.
