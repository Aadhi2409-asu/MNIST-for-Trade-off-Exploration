import numpy as np
import struct
from pathlib import Path
from sklearn.model_selection import train_test_split
# function of this code is to load the data from the file and return it as a numpy array
RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
def load_labels(file_path):
    print(f"Loading labels from {file_path}...")
    with open(file_path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Invalid magic number in label file: {magic}".format(magic))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels
def load_images(file_path):
    print(f"Loading images from {file_path}...")
    with open(file_path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Invalid magic number in image file: {}".format(magic))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows * cols)

    return images
def load_mnist():
    print("Loading MNIST dataset...")
    X_train = load_images(RAW_DATA_PATH / "train-images.idx3-ubyte")
    y_train = load_labels(RAW_DATA_PATH / "train-labels.idx1-ubyte")
    X_test = load_images(RAW_DATA_PATH / "t10k-images.idx3-ubyte")
    y_test = load_labels(RAW_DATA_PATH / "t10k-labels.idx1-ubyte")
    print("MNIST dataset loaded successfully.")
    return X_train, X_test, y_train, y_test

def validation_split(X, y, val_size=0.2, random_state=42):
    print(f"Splitting data into training and validation sets with validation size {val_size}...")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=val_size, random_state=random_state)
    print(f"Data split completed. Training set size: {len(X_tr)}, Validation set size: {len(X_val)}")
    return X_tr, X_val, y_tr, y_val

    
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_mnist()
    print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}, X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")
    X_train, X_val, y_train, y_val = validation_split(X_train, y_train)
    
    