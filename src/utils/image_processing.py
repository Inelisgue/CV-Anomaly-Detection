import cv2
import numpy as np

def preprocess_image(image_path, size=(128, 128)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0

def calculate_reconstruction_error(original, reconstructed):
    return np.mean((original - reconstructed)**2)
