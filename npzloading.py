import numpy as np

# Load the .npz file
filename = "./spec_repr/c/00_BN1-129-Eb_comp.npz"
loaded = np.load(filename)

# See what's inside
print("Keys in .npz:", list(loaded.keys()))
# Output: ['repr', 'labels']

# Access the arrays
repr_data = loaded["repr"]      # Shape: (num_frames, 192)
labels_data = loaded["labels"]  # Shape: (num_frames, 6, 21)

# Check shapes
print("Repr shape:", repr_data.shape)      # e.g., (1290, 192)
print("Labels shape:", labels_data.shape)  # e.g., (1290, 6, 21)

# Look at specific frame
frame_100_repr = repr_data[100]      # (192,) - CQT for frame 100
frame_100_labels = labels_data[100]  # (6, 21) - frets for frame 100

print("\nFrame 100 repr shape:", frame_100_repr.shape)
print("Frame 100 labels shape:", frame_100_labels.shape)

# Check values
print("\nFrame 100 repr values (first 10 bins):", repr_data[100, :10])
print("Frame 100 label for string 0:", labels_data[100, 0])  # (21,) one-hot
print("  Predicted fret class:", np.argmax(labels_data[100, 0]))  # which class has highest prob

# Access all strings for frame 100
print("\nAll 6 strings for frame 100:")
for i in range(6):
    fret_class = np.argmax(labels_data[100, i])
    print(f"  String {i}: fret class {fret_class}")