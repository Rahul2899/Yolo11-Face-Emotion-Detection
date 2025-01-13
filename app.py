from ultralytics import YOLO
import cv2
import sys

# Define a set of accessible colors for different labels
CLASS_COLORS = [
    (255, 0, 0),   # Blue
    (0, 255, 0),   # Green
    (0, 0, 255),   # Red
    (255, 255, 0), # Cyan
    (255, 0, 255), # Magenta
    (0, 255, 255), # Yellow
    (128, 128, 128), # Gray
]

def process_frame(frame, model):
    """
    Process a frame using the YOLO model and return the annotated frame with bounding boxes and labels.
    """
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    gray_image_3d = cv2.merge([gray_image, gray_image, gray_image])  # Convert to 3 channels
    results = model(gray_image_3d)
    result = results[0]

    # Draw bounding boxes and labels manually
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        conf = box.conf[0]  # Confidence score
        label_index = int(box.cls[0])  # Class index
        label = result.names[label_index]  # Class label
        color = CLASS_COLORS[label_index % len(CLASS_COLORS)]  # Cycle through colors

        # Draw a thicker bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Add the label and confidence
        label_text = f"{label} {conf:.2f}"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x, text_y = x1, y1 - 10  # Text position
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)  # Text background
        cv2.putText(frame, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame

def test_image(image_path, model):
    """
    Test the model on a single image and display the results.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    annotated_frame = process_frame(frame, model)
    cv2.imshow("YOLO Inference - Image", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_video(video_path, model):
    """
    Run inference on a video file and display the results in real time.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video at {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = process_frame(frame, model)
        cv2.imshow("YOLO Inference - Video", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

def test_webcam(model):
    """
    Run inference using the webcam and display the results in real time.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from the webcam.")
            break

        annotated_frame = process_frame(frame, model)
        cv2.imshow("YOLO Inference - Webcam", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load the YOLO model
    model_path = "best.onnx"  # Replace with your ONNX model path
    model = YOLO(model_path)

    print("Choose an option:")
    print("1. Test with an image")
    print("2. Test with a video file")
    print("3. Test with the webcam")

    choice = input("Enter your choice (1/2/3): ")

    if choice == "1":
        image_path = input("Enter the path to the image: ")
        test_image(image_path, model)
    elif choice == "2":
        video_path = input("Enter the path to the video file: ")
        test_video(video_path, model)
    elif choice == "3":
        test_webcam(model)
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
