import sys
sys.path.append(".")

import numpy as np
from pathlib import Path
from src.data.loader import load_mnist,validation_split

DATA_PROCESSED = Path("data/processed")


def normalize(X):
    print("Normalizing pixel values to [0, 1] range...")
    return X.astype(np.float32) / 255.0


def reshape_for_cnn(X):
    print("Reshaping data for CNN...")
    return X.reshape(-1, 28, 28, 1)


def save_processed(data: dict):
    print("Saving processed data to disk...")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, array in data.items():
        np.save(DATA_PROCESSED / f"{name}.npy", array)
    print("Done")


def load_processed():
    print("Loading processed data from disk...")
    keys = ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]
    data = {}
    for key in keys:
        file_path = DATA_PROCESSED / f"{key}.npy"
        if not file_path.exists():
            raise FileNotFoundError(f"{key}.npy not found. Run preprocessor.py first.")
        data[key] = np.load(file_path)
    print('Done')
    
    return data


if __name__ == "__main__":
    X_train_raw, X_test_raw, y_train_raw, y_test = load_mnist()
    X_train, X_val, y_train, y_val = validation_split(X_train_raw, y_train_raw)

    save_processed({
        "X_train": normalize(X_train),
        "X_val"  : normalize(X_val),
        "X_test" : normalize(X_test_raw),
        "y_train": y_train,
        "y_val"  : y_val,
        "y_test" : y_test,
    })

    load_processed()