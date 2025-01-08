import cv2
from utils.preprocess import preprocess_frame

def process_video(input_video_path, output_video_path, model, class_labels):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  

    # Initialize video writer
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame)

        # Predict using the model
        predictions = model.predict(preprocessed_frame)
        predicted_class = predictions.argmax(axis=1)[0]
        predicted_label = class_labels[predicted_class]

        # Annotate the frame
        cv2.putText(frame, f"Detected: {predicted_label}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Write the frame to output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output video saved at {output_video_path}")
