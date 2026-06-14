# ==========================================
# Part C - Image Data (CIFAR-10)
# ==========================================

import pickle
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------
# Function to load CIFAR batch
# ------------------------------------------
def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

# ------------------------------------------
# 1. Load an Image
# ------------------------------------------
batch = unpickle(r"D:\SEM_5\ML\datasets\Image_data\cifar-10-batches-py\data_batch_1")

# First image
image = batch[b'data'][0]

# ------------------------------------------
# Convert image format
# ------------------------------------------
image = image.reshape(3, 32, 32)
image = np.transpose(image, (1, 2, 0))

# ------------------------------------------
# 2. Display the Image
# ------------------------------------------
plt.imshow(image)
plt.title("CIFAR-10 Image")
plt.axis('off')
plt.show()

# ------------------------------------------
# 3. Determine Image Dimensions
# ------------------------------------------
height, width, channels = image.shape

print("Image Dimensions:")
print("Height =", height)
print("Width =", width)
print("Channels =", channels)

# ------------------------------------------
# 4. RGB or Grayscale
# ------------------------------------------
if channels == 3:
    print("\nImage Type: RGB")

    # Convert RGB to Grayscale
    gray = np.mean(image, axis=2)

    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis('off')
    plt.show()

else:
    print("\nImage Type: Grayscale")

# ------------------------------------------
# 5. Total Number of Pixels
# ------------------------------------------
pixels = height * width

print("\nTotal Number of Pixels =", pixels)