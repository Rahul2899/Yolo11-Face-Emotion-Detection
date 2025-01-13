# YOLO Object Detection with OpenCV

This project demonstrates how to use a YOLO model for object detection using images, videos, and a webcam in real-time. The script allows testing the model on different input types and visualizes bounding boxes, labels, and confidence scores on the output.

## Features

- **Image Inference:** Load an image file, detect objects, and display annotated results.
- **Video Inference:** Process a video file frame-by-frame, adding detections in real time.
- **Webcam Inference:** Use your webcam for live object detection.

The implementation leverages the [Ultralytics YOLO library](https://github.com/ultralytics/ultralytics) and OpenCV for visualization.

---

## Requirements

Before running the project, make sure the following dependencies are installed:

- Python 3.8 or later
- OpenCV
- Ultralytics YOLO

### Install the dependencies:

```bash
pip install ultralytics opencv-python
```

---

## Usage Instructions

1. **Prepare your YOLO Model:**

   - Replace the `model_path` in the script with the path to your ONNX YOLO model (e.g., `best.onnx`).

2. **Run the script:**

   Execute the script by running:

   ```bash
   python yolodemo.py
   ```

3. **Choose the mode of operation:**

   - `1`: Perform inference on an image file.
   - `2`: Perform inference on a video file.
   - `3`: Perform real-time inference using your webcam.

4. **Provide the input file (if applicable):**

   - For **image or video mode**, you’ll be prompted to enter the path to the file.

---

## Functions Overview

- `process_frame`: Processes a single frame, runs inference, and annotates the frame with bounding boxes, labels, and confidence scores.
- `test_image`: Runs inference on a single image file.
- `test_video`: Runs inference on a video file frame-by-frame.
- `test_webcam`: Performs live inference using the webcam.

---

## Key Highlights

- **Customizable Label Colors:** The script uses a fixed set of accessible colors for labels, cycling through if there are more classes than colors.
- **Real-time Processing:** Efficient processing allows for smooth performance on real-time webcam feeds.
- **ONNX Model Compatibility:** Leverages ONNX format for model flexibility across different frameworks.

---

## Troubleshooting

1. **Image/Video Not Found:**
   Ensure the file path provided is correct.

2. **Webcam Not Accessible:**
   Verify that your webcam is not being used by another application.

3. **Model Loading Issues:**
   Confirm that the specified model path exists and is compatible with the `YOLO` class.

4. **Dependencies Error:**
   Make sure all required libraries are installed (`ultralytics`, `opencv-python`).

---

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute it.

---

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for the YOLO library.
- [OpenCV](https://opencv.org/) for powerful computer vision tools.


