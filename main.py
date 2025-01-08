import os
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_frame
from utils.class_labels import get_class_labels
from process_video import process_video

MODEL_PATH = "models/traffic_sign_model.h5"
VIDEO_PATH = "data/video.mp4"
OUTPUT_PATH = "data/output_video.mp4"

def main():
    # Load the trained model
    model = load_model(MODEL_PATH)
    
    # Load class labels
    class_labels = get_class_labels()

    # Process the video
    process_video(VIDEO_PATH, OUTPUT_PATH, model, class_labels)
    print(f"Output video saved at {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
