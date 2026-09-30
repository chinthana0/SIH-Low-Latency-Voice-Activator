import os
import numpy as np

from preprocess import load_audio, extract_mfcc


def load_dataset(dataset_path):
    X = []
    y = []

    # Positive = wake word
    # Negative + background = should NOT trigger
    classes = [
        ("positive_train", 1),
        ("negative_train", 0),
        ("background_train", 0),
    ]

    for folder_name, label in classes:

        folder = os.path.join(dataset_path, folder_name)

        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue

        print(f"Loading: {folder_name}")

        for filename in os.listdir(folder):

            if filename.lower().endswith(".wav"):

                file_path = os.path.join(folder, filename)

                try:
                    audio = load_audio(file_path)
                    mfcc = extract_mfcc(audio)

                    X.append(mfcc)
                    y.append(label)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    return X, y


if __name__ == "__main__":

    dataset_path = r"C:\Users\chint\Downloads\output-20260926T115017Z-1-001\output\hey_mist"

    X, y = load_dataset(dataset_path)

    print("\nDataset loaded!")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Positive samples:", np.sum(y == 1))
    print("Negative samples:", np.sum(y == 0))
