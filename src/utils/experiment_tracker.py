import json
import os
from datetime import datetime

LOG_PATH = os.path.join("outputs", "experiment_log.json")


def load_log():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_log(log):
    os.makedirs("outputs", exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


def log_experiment(config, metrics, notes=""):
    log = load_log()

    version = len(log) + 1

    entry = {
        "version": version,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": notes,
        "config": {
            "epochs_run":      config.get("epochs_run"),
            "seq_length":      config.get("seq_length"),
            "batch_size":      config.get("batch_size"),
            "embed_dim":       config.get("embed_dim"),
            "lstm_units":      config.get("lstm_units"),
            "num_layers":      config.get("num_layers"),
            "dropout":         config.get("dropout"),
            "learning_rate":   config.get("learning_rate"),
        },
        "metrics": {
            "train": {
                "loss":       float(metrics["train"]["loss"]),
                "perplexity": float(metrics["train"]["perplexity"]),
                "accuracy":   float(metrics["train"]["accuracy"]),
            },
            "val": {
                "loss":       float(metrics["val"]["loss"]),
                "perplexity": float(metrics["val"]["perplexity"]),
                "accuracy":   float(metrics["val"]["accuracy"]),
            },
            "test": {
                "loss":       float(metrics["test"]["loss"]),
                "perplexity": float(metrics["test"]["perplexity"]),
                "accuracy":   float(metrics["test"]["accuracy"]),
            },
            "train_val_gap":  float(metrics["val"]["loss"] - metrics["train"]["loss"]),
        }
    }

    log.append(entry)
    save_log(log)
    print(f"\nExperiment v{version} logged to {LOG_PATH}")
    return version