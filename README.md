# PCB Inspect

An automated Printed Circuit Board (PCB) defect detection system using YOLOv11 deep learning model. This system can identify and classify six common PCB defects through a web-based interface.

## Overview

This project implements a real-time PCB defect detection system that uses computer vision and deep learning to automatically identify manufacturing defects in PCB images. The system is built with YOLOv11 (Ultralytics) and provides both a web interface and REST API for defect detection.

## Detected Defect Types

The system can detect the following six types of PCB defects:

1. **Missing Hole** - Missing drill holes in the PCB (CRITICAL)
2. **Mouse Bite** - Irregular edges caused by depaneling (WARNING)
3. **Open Circuit** - Broken or disconnected circuit traces (CRITICAL)
4. **Short** - Unintended connections between circuits (CRITICAL)
5. **Spur** - Excess copper or material spurs (WARNING)
6. **Spurious Copper** - Unwanted copper deposits (WARNING)

Defects are categorized by severity:
- **CRITICAL**: Failures that render the PCB non-functional
- **WARNING**: Issues that may cause problems but don't immediately fail the board

## Model Performance

### Overall Metrics (Test Set: 35 images)

| Metric | Value |
|--------|-------|
| **Precision** | 0.952 (95.2%) |
| **Recall** | 0.88 (88.0%) |
| **mAP@50** | 0.92 (92.0%) |
| **mAP@50-95** | 0.494 (49.4%) |

### Per-Class Performance

| Defect Type | Precision | Recall | mAP@50 | mAP@50-95 |
|-------------|-----------|--------|--------|-----------|
| Missing Hole | 1.000 | 0.946 | 0.978 | 0.540 |
| Open Circuit | 0.977 | 0.895 | 0.990 | 0.581 |
| Short | 0.942 | 0.966 | 0.988 | 0.564 |
| Spur | 0.956 | 0.826 | 0.890 | 0.436 |
| Mouse Bite | 0.909 | 0.812 | 0.818 | 0.437 |
| Spurious Copper | 0.926 | 0.836 | 0.856 | 0.408 |

### Dataset Statistics

- **Training Set**: 526 images
- **Validation Set**: 132 images
- **Test Set**: 35 images
- **Total Images**: 693 images
- **Training Epochs**: 200
- **Model Architecture**: YOLOv11n (Nano)

## Features

- **Image Upload Interface**: User-friendly web interface for uploading PCB images
- **Real-time Detection**: Fast inference using YOLOv11 optimized model
- **Detailed Results**: Bounding boxes, confidence scores, and defect classifications
- **Visual Annotations**: Annotated images with highlighted defect regions
- **Severity Assessment**: Automatic verdict assignment (PASS/MARGINAL/FAIL)
- **REST API**: Programmatic access via HTTP endpoints
- **Batch Processing**: Support for processing multiple images
- **GPU Support**: Automatic CUDA detection for faster inference

## Quick Start

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional, for faster inference)
- pip or uv package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd PCB_Defect_Detection
   ```

2. **Install dependencies**

   Using pip:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or using uv:
   ```bash
   uv pip install -e .
   ```

3. **Download model weights**

   The trained model weights should be placed in the `models/` directory:
   - `models/best.pt` - Main model file (should be present)

### Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```
   Or:
   ```bash
   flask run
   ```

2. **Access the web interface**
   
   Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

3. **Upload a PCB image**
   
   - Click "Choose File" to select a PCB image (JPG, JPEG, or PNG)
   - Click "Upload and Detect" to process the image
   - View the detection results with bounding boxes and classifications

## Project Structure

```
PCB_Defect_Detection/
├── app.py                 # Main Flask application
├── main.py                # Entry point (optional)
├── pyproject.toml         # Project dependencies
├── README.md              # This file
├── LICENSE                # License file
│
├── models/                # Model weights
│   ├── best.pt           # Trained YOLOv11 model
│   └── yolo11n.pt        # Base YOLOv11n weights
│
├── src/                   # Source code
│   ├── config/
│   │   └── settings.py   # Configuration settings
│   ├── model/
│   │   └── predictor.py  # Model inference class
│   └── utils/
│       └── file_handler.py # File handling utilities
│
├── templates/             # HTML templates
│   └── index.html        # Main web interface
│
├── static/                # Static files
│   ├── style.css         # Stylesheet
│   └── main.js           # Frontend JavaScript
│
├── data/                  # Data directories
│   ├── uploads/          # Uploaded images
│   └── results/          # Detection results
│
└── test/                  # Testing and evaluation
    ├── dataset/          # Test dataset
    └── PCB_Defect_detection.ipynb  # Evaluation notebook
```

## API Endpoints

### Upload and Detect
```http
POST /upload
Content-Type: multipart/form-data

file: <image file>
```

**Response:**
```json
{
  "success": true,
  "filename": "20260104_123456_pcb_image.jpg",
  "annotated_filename": "annotated_20260104_123456_pcb_image.jpg",
  "verdict": "FAIL",
  "num_detections": 3,
  "defect_summary": {
    "Open_circuit": {
      "count": 2,
      "severity": "CRITICAL",
      "avg_confidence": 0.87
    },
    "Spur": {
      "count": 1,
      "severity": "WARNING",
      "avg_confidence": 0.92
    }
  },
  "detections": [...],
  "timestamp": "2024-01-04T12:34:56.789Z"
}
```

### Predict (JSON)
```http
POST /predict
Content-Type: application/json

{
  "image_path": "/path/to/image.jpg",
  "save_output": true
}
```

### Batch Predict
```http
POST /batch-predict
Content-Type: application/json

{
  "image_paths": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
  "save_outputs": true
}
```

### Configuration
```http
GET /config        # Get current configuration
POST /config       # Update configuration
```

### Model Info
```http
GET /model-info    # Get model information
```

### Health Check
```http
GET /health        # System health status
```

## Configuration

Configuration can be set via environment variables or by modifying `src/config/settings.py`:

- `MODEL_PATH`: Path to model weights (default: `models/best.pt`)
- `CONF_THRESHOLD`: Confidence threshold for detections (default: `0.25`)
- `IOU_THRESHOLD`: IoU threshold for NMS (default: `0.45`)
- `UPLOAD_FOLDER`: Directory for uploaded files (default: `data/uploads`)
- `RESULTS_FOLDER`: Directory for results (default: `data/results`)
- `PORT`: Server port (default: `5000`)
- `HOST`: Server host (default: `0.0.0.0`)
- `DEBUG`: Debug mode (default: `False`)

## Model Training

The model was trained using the Ultralytics YOLOv11 framework:

1. **Dataset Preparation**: 
   - Converted XML annotations to YOLO format
   - Split data into train/val/test sets (75.9%/19.0%/5.1%)
   
2. **Training Configuration**:
   - Base Model: YOLOv11n (Nano)
   - Epochs: 200
   - Image Size: 640x640
   - Batch Size: 16
   - Learning Rate: 0.01

3. **Evaluation**:
   - See `test/PCB_Defect_detection.ipynb` for detailed evaluation metrics
   - Metrics calculated on held-out test set

## Verdict Logic

The system assigns one of three verdicts:

- **PASS**: No defects detected
- **MARGINAL**: Warning-level defects detected (non-critical issues)
- **FAIL**: Critical defects detected (Open Circuit, Short, Missing Hole)

## Development

### Running Tests
```bash
# Run evaluation notebook
jupyter notebook test/PCB_Defect_detection.ipynb
```

### Code Structure
- `PCBDefectPredictor`: Main model inference class
- `Config`: Application configuration
- Flask routes: API endpoints and web interface


## Acknowledgments

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) for the detection framework
- PCB defect dataset used for training

---

**Note**: This system is designed for quality control in PCB manufacturing. Always verify critical detections with human inspection.
