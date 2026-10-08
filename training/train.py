"""Train a tiny CNN for ASL alphabet recognition (A-Z + 'nothing').

Input : 64x64 grayscale image, float in [0, 1]
Output: 27-way softmax
Artifacts are written to training/out/.
"""
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data" / "asl_alphabet_train" / "asl_alphabet_train"
OUT = Path(__file__).resolve().parent / "out"
CACHE = OUT / "dataset_64.npz"

IMG = 64
LABELS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["nothing"]
SEED = 42
EPOCHS = 15
BATCH = 64


def load_dataset():
    if CACHE.exists():
        d = np.load(CACHE)
        return d["x"], d["y"]
    xs, ys = [], []
    for idx, label in enumerate(LABELS):
        files = sorted((TRAIN_DIR / label).glob("*.jpg"))
        print(f"{label}: {len(files)} images")
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            # Same luma weights the Android app uses (0.299R + 0.587G + 0.114B).
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            xs.append(cv2.resize(gray, (IMG, IMG), interpolation=cv2.INTER_AREA))
            ys.append(idx)
    x = np.stack(xs).astype(np.uint8)
    y = np.array(ys, dtype=np.int64)
    np.savez_compressed(CACHE, x=x, y=y)
    return x, y


def split(x, y):
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(x, y, test_size=0.2, stratify=y, random_state=SEED)
    x_va, x_te, y_va, y_te = train_test_split(x_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)
    return (x_tr, y_tr), (x_va, y_va), (x_te, y_te)


def to_float(x):
    return (x.astype(np.float32) / 255.0)[..., None]


augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),        # left/right-handed signers + mirrored front camera
    tf.keras.layers.RandomRotation(15 / 360),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomBrightness(0.25, value_range=(0.0, 1.0)),
    tf.keras.layers.RandomContrast(0.3),
], name="augment")


def make_ds(x, y, training):
    ds = tf.data.Dataset.from_tensor_slices((to_float(x), y))
    if training:
        ds = ds.shuffle(10000, seed=SEED)
    ds = ds.batch(BATCH)
    if training:
        ds = ds.map(lambda a, b: (tf.clip_by_value(augment(a, training=True), 0.0, 1.0), b),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model():
    inp = tf.keras.Input((IMG, IMG, 1), name="image")
    x = inp
    for filters in (8, 16, 32, 64):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(96, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(len(LABELS), activation="softmax", name="probs")(x)
    return tf.keras.Model(inp, out, name="tiny_asl_cnn")


def main():
    OUT.mkdir(exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)
    x, y = load_dataset()
    print("Dataset:", x.shape, "classes:", len(LABELS))
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = split(x, y)
    np.savez_compressed(OUT / "test_split.npz", x=x_te, y=y_te)
    np.savez_compressed(OUT / "rep_split.npz", x=x_tr[:500])

    model = build_model()
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(
        make_ds(x_tr, y_tr, True), validation_data=make_ds(x_va, y_va, False), epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
        ],
    )
    model.save(OUT / "asl_fp32.keras")
    (OUT / "labels.txt").write_text("\n".join(LABELS) + "\n")

    loss, acc = model.evaluate(make_ds(x_te, y_te, False), verbose=0)
    y_pred = model.predict(to_float(x_te), batch_size=256, verbose=0).argmax(1)
    print(f"Test accuracy (fp32): {acc:.4f}")
    report = classification_report(y_te, y_pred, target_names=LABELS, digits=4)
    print(report)
    (OUT / "classification_report_fp32.txt").write_text(report)
    json.dump({"test_acc_fp32": float(acc), "params": int(model.count_params()),
               "epochs_run": len(hist.history["loss"])},
              open(OUT / "train_metrics.json", "w"), indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k in ("accuracy", "loss"):
        a = ax[0] if k == "accuracy" else ax[1]
        a.plot(hist.history[k], label="train")
        a.plot(hist.history["val_" + k], label="val")
        a.set_title(k); a.set_xlabel("epoch"); a.legend()
    fig.tight_layout(); fig.savefig(OUT / "training_curves.png", dpi=120)

    fig, ax = plt.subplots(figsize=(12, 12))
    ConfusionMatrixDisplay(confusion_matrix(y_te, y_pred), display_labels=LABELS).plot(
        ax=ax, colorbar=False, xticks_rotation=90)
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrix_fp32.png", dpi=120)


if __name__ == "__main__":
    main()
