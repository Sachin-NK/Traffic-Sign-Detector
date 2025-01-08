import cv2
import numpy as np

def preprocess_frame(frame, img_size=(441, 441)):
    resized_frame = cv2.resize(frame, img_size)  # Resize to input size
    normalized_frame = resized_frame / 255.0    # Normalize pixel values
    return np.expand_dims(normalized_frame, axis=0)  # Add batch dimension
