import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load labels from labels.txt
with open('labels.txt', 'r') as file:
    labels = file.read().splitlines()

# Path setup
base_path = 'C:\\gesture_interpretation1\\sld10\\'
sequence_length = 30  # Number of frames per sequence
total_sequences = 30  # Total sequences to capture

# Prepare the dataset
X_train_frames = []
X_train_features = []
y_train = []
X_test_frames = []
X_test_features = []
y_test = []

for label in labels:
    for sequence_count in range(total_sequences):
        frames_path = os.path.join(base_path, label, f"{label}_seq{sequence_count}", f"{label}_seq{sequence_count}_frames.npy")
        features_path = os.path.join(base_path, label, f"{label}_seq{sequence_count}", f"{label}_seq{sequence_count}_features.npy")

        if not os.path.isfile(frames_path) or not os.path.isfile(features_path):
            print(f"Warning: File for label '{label}' sequence {sequence_count} does not exist. Skipping this label.")
            continue

        frames = np.load(frames_path)
        features = np.load(features_path)
        if len(frames) < sequence_length or len(features) < sequence_length:
            print(f"Warning: Not enough frames or features in '{frames_path}' or '{features_path}' to create sequences.")
            continue

        if sequence_count < int(total_sequences * 0.8):  # Use 80% for training
            X_train_frames.append(frames)
            X_train_features.append(features)
            y_train.append(label)
        else:  # Use 20% for testing
            X_test_frames.append(frames)
            X_test_features.append(features)
            y_test.append(label)

X_train_frames = np.array(X_train_frames)
X_train_features = np.array(X_train_features)
y_train = np.array(y_train)
X_test_frames = np.array(X_test_frames)
X_test_features = np.array(X_test_features)
y_test = np.array(y_test)

if len(y_train) == 0 or len(y_test) == 0:
    print("Error: No valid sequences found.")
    exit()

unique_labels, y_train_encoded = np.unique(y_train, return_inverse=True)
_, y_test_encoded = np.unique(y_test, return_inverse=True)
y_train_encoded = to_categorical(y_train_encoded, num_classes=len(unique_labels))
y_test_encoded = to_categorical(y_test_encoded, num_classes=len(unique_labels))

np.save('X_train_frames.npy', X_train_frames)
np.save('X_train_features.npy', X_train_features)
np.save('y_train.npy', y_train_encoded)
np.save('X_test_frames.npy', X_test_frames)
np.save('X_test_features.npy', X_test_features)
np.save('y_test.npy', y_test_encoded)
np.save('unique_labels.npy',unique_labels)

print(f"Saved dataset as 'X_train_frames.npy', 'X_train_features.npy', 'y_train.npy', 'X_test_frames.npy', 'X_test_features.npy', and 'y_test.npy'.")
print(f"Unique labels found: {unique_labels}")