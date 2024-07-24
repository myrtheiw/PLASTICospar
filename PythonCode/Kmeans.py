# K-means
import cv2
import numpy as np
from sklearn.cluster import KMeans
import glob
import matplotlib.pyplot as plt

# Step 1: Collect Pixel Data from Multiple Images
def collect_pixels(image_paths):
    pixels = []
    for path in image_paths:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels.append(image.reshape(-1, 3))
    return np.vstack(pixels)

# List of image paths
image_paths = glob.glob('Data/*.jpg') 

# Collect pixels from all images
all_pixels = collect_pixels(image_paths)

# Step 2: Train K-Means on Combined Data
k = 100  # Adjust based on the number of clusters you expect
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(all_pixels)

# Function to apply the trained K-Means model to an image
def segment_image(image_path, kmeans_model):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)
    labels = kmeans_model.predict(pixels)
    segmented_image = labels.reshape(image.shape[0], image.shape[1])
    return segmented_image

# Step 3: Apply the Trained K-Means Model to Each Image
for path in image_paths:
    segmented_image = segment_image(path, kmeans)
    plt.figure()
    plt.imshow(segmented_image, cmap='viridis')
    plt.title(f'Segmented Image: {path}')
    plt.axis('off')
    plt.show()
