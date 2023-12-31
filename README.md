# Automatic Number Plate Recognition (ANPR) using YOLOv8 and EasyOCR

This project is an Automatic Number Plate Recognition (ANPR) system that utilizes the YOLOv8 object detection model and EasyOCR library to detect and extract text from vehicle number plates in images.

## Features

- Detects vehicles and localizes number plates using YOLOv8.
- Extracts text from number plates using EasyOCR.
- Supports various image formats (e.g., JPEG, PNG).
- Provides easy-to-use interface for interacting with the ANPR system.

## Prerequisites

Make sure you have the following dependencies installed:

- Python (version >= 3.10)
- OpenCV (cv2)
- Matplotlib
- EasyOCR
- PyTorch
- NumPy
- Ultralytics YOLO

## Installation

1. Clone the repository:

```shell
git clone https://github.com/Amr2087/Automatic-Number-Plate-Recognition.git
```

2. Install the required Python dependencies:

```shell
pip install -r requirements.txt
```

## Usage

1. Place the input images containing vehicles in the `images` directory.
2. Open the notebook
3. intiate the model
4. use the predict function
5. The script will process the images and output the detected number plates along with the extracted text.

## Configuration

- The YOLOv8 model configuration and weights can be customized to achieve desired accuracy and performance.

## Contributing

Contributions to this project are welcome! If you encounter any issues or have suggestions for improvements, please submit them as GitHub issues.

## Acknowledgments

- The YOLOv8 model used in this project is based on the work of [Ultralytics](https://github.com/ultralytics/yolov8).
- EasyOCR library is developed by [JaidedAI](https://github.com/JaidedAI/EasyOCR).

## Contact

For any inquiries or questions, please contact [amrahmedfarouk70@gmail.com](mailto:your-email@example.com).

Feel free to update the sections, add more information, and provide additional details that are relevant to your project. Including example outputs, screenshots, or diagrams can also be helpful for users to understand the project better.
