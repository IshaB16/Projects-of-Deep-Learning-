import argparse, json
from pathlib import Path
import tensorflow as tf
from src.common import load_datasets, build_model

def main():
    p = argparse.ArgumentParser(description="Train MobileNetV2 chest X-ray classifier")
    p.add_argument("--data-dir", default="data/chest_xray")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--weights", default="imagenet", help="imagenet or None")
    p.add_argument("--fine-tune", action="store_true")
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1:
        p.error("--epochs and --batch-size must be positive")
    train, val, _ = load_datasets(a.data_dir, a.batch_size)
    if a.max_train_batches: train = train.take(a.max_train_batches)
    if a.max_val_batches: val = val.take(a.max_val_batches)
    model = build_model(a.weights, a.learning_rate, a.dropout, a.fine_tune)
    Path("models").mkdir(exist_ok=True); Path("reports").mkdir(exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("models/pneumonia_model.keras", monitor="val_auc",
            mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger("reports/training_log.csv", append=True)]
    history = model.fit(train, validation_data=val, epochs=a.epochs, callbacks=callbacks)
    model.save("models/pneumonia_model.keras")
    Path("reports/training_history.json").write_text(
        json.dumps({k: [float(x) for x in v] for k, v in history.history.items()}, indent=2),
        encoding="utf-8")
    print("Saved model to models/pneumonia_model.keras")

if __name__ == "__main__":
    main()
