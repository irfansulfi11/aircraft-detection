# 🛰️ AeroVision - SAR Aircraft Detection System

AeroVision is a web-based AI tool designed to detect aircraft in Synthetic Aperture Radar (SAR) images using the YOLOv8 object detection model. Built with Flask, this sleek application provides a modern glassmorphism interface for real-time inference and visual analytics.

---

## ✨ Features

- 🧠 YOLOv8-powered aircraft detection
- 📤 Drag-and-drop SAR image upload
- 🎯 Bounding box visualization with confidence scores
- 📊 Stats dashboard: Detection count, average confidence, inference time, threat level
- 🖼️ Inline display of original & annotated results
- ⬇️ Download support for original & result images
- 🌐 Fully responsive design with a modern white & steel blue color palette
- 🔒 Secure upload handling with unique filenames

---

## 📁 Project Structure

```
SAR-aircraft/
├── app.py                  # Flask backend with YOLOv8 integration
├── runs/                  # Contains trained YOLO weights (best.pt)
│   └── detect/sar_aircraft_detector/weights/best.pt
├── uploads/               # Uploaded SAR images (auto-created)
├── results/               # Detection result images (auto-created)
├── static/                # Future CSS or JS files
└── templates/
    └── index.html         # Frontend HTML UI (auto-generated if missing)
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sar-aircraft-detection.git
cd sar-aircraft-detection
```

### 2. Install Dependencies

Make sure you have Python 3.8 or later.

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install flask ultralytics opencv-python
```

### 3. Download YOLOv8 Weights

Ensure your trained YOLOv8 model (`best.pt`) is placed at:

```
runs/detect/sar_aircraft_detector/weights/best.pt
```

You can train your model using:
```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

### 4. Run the App

```bash
python app.py
```

Open your browser and navigate to:

```
http://localhost:5000
```

---

## 🧪 Supported File Types

- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.tif`, `.tiff`

---

## 📦 API Endpoint

### `POST /upload`

**Request:**  
Form-data with a `file` field containing the image.

**Response:**  
```json
{
  "success": true,
  "original_image": "<base64>",
  "result_image": "<base64>",
  "detections": [
    {
      "confidence": 0.92,
      "class_id": 0,
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "detection_count": 3,
  "filename": "20250717_134512_sample.jpg"
}
```

---

## 📸 UI Preview

![Preview Screenshot](https://placehold.co/800x400?text=AeroVision+Detection+UI)

---

## 🔐 Security Notes

- Filenames are sanitized using `secure_filename`
- Secret key is randomly generated using `secrets.token_hex()`
- Model loads only once on server startup to avoid delay

---

## 🛠️ Future Improvements

- Add user authentication (admin access)
- Historical analysis and result log
- PDF/CSV export of detection data
- Dark/light theme toggle
- Map-based visualization for geospatial SAR inputs


## 🙌 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Flask](https://flask.palletsprojects.com/)
- Open-source SAR datasets for training

---

> Built with 🛡️ for defense intelligence and aerospace innovation.
