# Experiment 8 – Hardware Deployment of a TinyML Model on an Edge Device

**Application:** Real-time American Sign Language (ASL) alphabet recognition (A–Z + "nothing")
**Edge device:** Android smartphone (on-device TensorFlow Lite int8 inference, no cloud)

## 1. Why this design (model selection)

We first considered the Hugging Face model `prithivMLmods/Alphabet-Sign-Language-Detection`. It is a fine-tuned SigLIP2-base vision transformer.

| Model | Params | Size | Fits an MCU (≤ 4 MB flash / ≤ 520 KB SRAM)? |
|---|---|---|---|
| SigLIP2-base (HF) | 92.9 M | ~370 MB fp32 / ~95 MB int8 | No, about 25× over flash even after int8 |
| **Our Tiny CNN** | 82,755 | **92.3 KB int8** | **Yes** |

TinyML models are typically 10–250 KB, so we designed and trained our own tiny CNN instead.

## 2. Pipeline

`Camera frame (640×480 RGBA)` → `centre ROI crop (70%)` → `rotate upright` → `64×64 grayscale` → `int8 quantise (scale/zero-point)` → `Interpreter.run()` → `dequantise → probabilities` → `threshold ≥ 0.80 held for 5 frames` → **torch pulse + vibration + letter appended**

| Lab-sheet step (TFLM on MCU) | This implementation (Android) |
|---|---|
| `model_data.h` C byte-array in flash | `asl_int8.tflite` asset, stored uncompressed and memory-mapped. `model_data.h/.cc` are also generated for MCU use. |
| Tensor arena (static `uint8_t[]`) | Interpreter native buffers. Measured as the native-heap delta at interpreter creation. |
| `MicroInterpreter` + `AllocateTensors()` | `Interpreter(model)` + `allocateTensors()` in `SignClassifier` init |
| `loop()` reading the sensor | CameraX `ImageAnalysis` analyzer (`STRATEGY_KEEP_ONLY_LATEST`) |
| `input(0)` / `Invoke()` / `output(0)` | int8 `ByteBuffer` / `interpreter.run()` / int8 output array |
| `millis()` / `micros()` | `SystemClock.elapsedRealtimeNanos()` |
| LED / piezo buzzer | Camera torch (LED flash) + vibration motor |

## 3. Training results (PC)

Dataset: Kaggle *ASL Alphabet* (grassknoted). We use 27 classes with 3000 images each, preprocessed to 64×64 grayscale and split 80/10/10 (stratified, seed 42). Augmentation: flip, ±15° rotation, zoom, translation, brightness, contrast.

| Model | Size (KB) | Test accuracy | PC latency / image (ms) |
|---|---|---|---|
| Keras fp32 | 1055.0 | 95.69% | – |
| TFLite fp32 | 326.8 | 95.69% | 0.169 |
| **TFLite int8** | **92.3** | **95.78%** | 0.266 |

Artefacts are in `training/out/`: `training_curves.png`, `confusion_matrix_fp32.png`, `classification_report_fp32.txt`, `quant_report.json`.

## 4. Measurement on the edge device (fill from app overlay / `adb logcat -s EdgeProfile`)

| Parameter | Measurement on Edge Device |
|---|---|
| Microcontroller Platform | Android phone: `<model>`, SoC `<e.g. Snapdragon 695>`, `<RAM>` GB, CPU, 2 threads, XNNPACK |
| Sensor Used | Rear camera (CMOS image sensor), 640×480 stream, 64×64 grayscale ROI |
| Flash Memory Used (KB / %) | Model 92.3 KB; APK `<x.xx>` MB (`<y>`% of `<storage>`) |
| SRAM / Tensor Arena Used (KB) | Interpreter native +`<n>` KB; total native heap `<n>` MB |
| Real-time Inference Latency (ms) | Invoke avg `<x>` ms, p95 `<x>` ms; full frame `<x>` ms; `<x>` FPS |
| Observed Accuracy/Responsiveness | Test set 95.78%. Live: `<k>/26` letters correct on first try (plain background, good light); actuation about `<x>` ms after the sign is held |

**How to read the live numbers:** after 300+ frames, read the overlay values: `Model`, `APK`, `Interp`, `Invoke avg/p95`, `FPS`. Alternatively, run:

```
adb logcat -s EdgeProfile
```

## 5. Observations

- int8 quantisation shrank the model 3.5× versus TFLite fp32 with an accuracy change of +0.09 percentage points (no loss).
- On the x86 PC, int8 was slightly *slower* than fp32 (0.27 vs 0.17 ms), because desktop CPUs and XNNPACK are heavily optimised for float. On ARM phones and MCUs, int8 is the faster path (NEON/SIMD integer kernels, no FPU needed), which is why latency must be measured on the target device.
- Weakest classes: V (F1 0.88), S (0.91), X (0.90), U (0.90), K (0.92), N (0.92). These are visually similar hand shapes at 64×64 grayscale.
- Real-world accuracy is lower than test accuracy. The Kaggle images come from one signer against one background and the split is random, so near-duplicate frames inflate the test score. Lighting, background clutter, skin tone, hand distance and motion blur all lower live confidence.
- Similar hand shapes get confused (M/N/S/T/A, U/V, and the motion letters J and Z).
- The 5-frame hold plus the 0.80 threshold removes most flicker and false actuations. The `nothing` class prevents firing on an empty ROI.

## 6. Post-lab questions

**Q1. Why is on-device inference preferable to streaming raw sensor data to the cloud?**
- **Latency:** a sign is classified in a few milliseconds locally. A cloud round-trip adds 100+ ms plus network jitter.
- **Bandwidth and energy:** streaming 640×480 video at 15–30 FPS costs megabits per second and radio power. On-device inference transmits nothing, or only the predicted letter.
- **Privacy:** camera frames of a person, or raw audio, never leave the device.
- **Reliability:** the system works offline, with no dependence on connectivity or a server.
- **Cost and scalability:** there is no per-inference server cost.

**Q2. What happens if the tensor arena + model weights + frame buffers exceed the available RAM?**
On an MCU, `AllocateTensors()` fails (`kTfLiteError`, "Arena size is too small"), or static allocation fails at link time. If memory is silently overrun, the stack or heap collides with other data. This causes hard faults, watchdog resets, corrupted outputs or random crashes. The only fixes are to:
- shrink the model (fewer filters, int8)
- lower the input resolution
- reuse or overlap buffers
- move data to PSRAM
- choose a larger device

On Android, the equivalent symptom is OutOfMemoryError, the app being killed by the low-memory killer, or heavy GC causing latency spikes.

**Q3. How do real-world environmental factors affect accuracy vs the test set?**
The test set comes from the same distribution as the training set. The real world adds domain shift:
- lighting changes and shadows
- cluttered backgrounds
- different hands and skin tones
- camera angle and distance
- motion blur, sensor noise and auto-exposure

For audio and IMU models, the equivalents are ambient noise and mounting position. Real accuracy therefore drops, as we observed. Mitigations: data augmentation, collecting data on the target sensor, an ROI or plain background, and confidence thresholds with temporal smoothing.

**Q4. Function of `AllocateTensors()` in the TFLM lifecycle.**
It runs after the interpreter is built from the model and op resolver, and before the first `Invoke()`. It:
- walks the model graph
- computes the size and lifetime of every intermediate tensor
- plans memory so that tensors whose lifetimes don't overlap share space
- carves all of them, plus the persistent buffers, out of the user-provided tensor arena
- calls each kernel's `Prepare`
- sets up the input/output tensor pointers

It is called once in `setup()`. It is also how you find the minimum arena size, via `arena_used_bytes()`.

## 7. Viva questions

1. **RAM and clock speed of TinyML targets:** typically 2 KB – 512 KB SRAM (a few MB with PSRAM), 32 KB – 4 MB flash, and clocks from tens of MHz to about 240 MHz (Cortex-M0+ to M7, ESP32). Power is in the mW range.
2. **Why convert the quantized model to a C byte array?** MCUs usually have no file system. Compiling the `.tflite` flatbuffer into firmware as a `const unsigned char[]` places it in flash. TFLM can then read weights in place, without copying to RAM. The array is aligned (`alignas(16)`) for safe access.
3. **The three memory budgets:**
   1. model weights in **flash**
   2. **tensor arena** in SRAM (activations, scratch, interpreter state)
   3. **input/frame buffer** for sensor data (camera frame, audio window, IMU window)
4. **How sensor resolution affects memory and latency:** activation memory and MACs scale with input pixels. Going from 64×64 to 96×96 is about 2.25× more compute and arena, and RGB vs grayscale is 3× the first-layer input and frame buffer. A higher audio sample rate or a longer window enlarges the spectrogram. Higher resolution can improve accuracy, but it costs RAM, latency and energy.
5. **The role of the LED / buzzer:** it closes the loop from inference to the physical world. It gives an immediate, observable confirmation of a prediction without a serial monitor, and lets you judge responsiveness and false-trigger rate by eye. It is also the actual output of the end-to-end Edge AI system. Here, the torch and vibration motor play that role.

## 8. Conclusion
*(Write in your own words.)* Suggested points:
- A 64×64 grayscale tiny CNN quantised to int8 (92.3 KB) reached 95.78% test accuracy.
- It runs fully on-device at `<x>` ms per inference and `<x>` FPS, and it drives a physical actuator.
- The 370 MB SigLIP2 model is unusable on edge hardware.
- The test vs real-world accuracy gap shows why deploying on the target sensor matters.
