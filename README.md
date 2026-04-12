# 🚦 Automated Traffic Management & ANPR System

[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-orange)](https://github.com/ultralytics/ultralytics)
[![Tesseract OCR](https://img.shields.io/badge/Tesseract-OCR-green)](https://github.com/tesseract-ocr/tesseract)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent computer vision application designed to automate traffic enforcement. By combining **YOLOv8** for high-speed object detection and **Tesseract OCR** for Automatic Number Plate Recognition (ANPR), this system detects moving vehicles, calculates their speed, and logs the license plates of those violating speed limits.

---

## ⚙️ How It Works (System Pipeline)

1. **Video Ingestion:** The system captures video frames (via live feed or `.mp4` files).
2. **Vehicle Tracking:** The generalized `yolov8n.pt` model detects and tracks vehicles across frames, assigning unique IDs.
3. **Speed Estimation:** By calculating the distance traveled over time between frames, the system estimates the velocity of each vehicle.
4. **ANPR Trigger:** If a vehicle exceeds the predefined speed threshold, the secondary `license_plate_detector.pt` model is triggered to isolate the vehicle's plate.
5. **Data Extraction:** The cropped license plate image is processed using Tesseract OCR to extract the alphanumeric text.

---

## 📦 Repository Structure

```text
Traffic-Management-System/
├── Traffic_Monitoring_Program.py         # Main execution script
├── yolov8n.pt                            # Pre-trained YOLOv8 nano model (Vehicles)
├── license_plate_detector.pt             # Custom trained model for plate detection
├── tesseract-ocr-w64-setup-...exe        # Windows Tesseract installer
├── requirements.txt                      # Python dependencies
├── traffic2.mp4                          # Sample video for testing
└── README.md                             # Project documentation
```
---

## 🚀Installation & Setup

### 1. Clone the Repository
~~~
git clone [https://github.com/MokshSharma2006/Traffic-Management-System.git](https://github.com/MokshSharma2006/Traffic-Management-System.git)

cd Traffic-Management-System
~~~

### 2. Set Up a Virtual Environment (Recommended)

~~~
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate

~~~

### 3. Install Dependencies

~~~
pip install -r requirements.txt
~~~

### 4. Install Tesseract OCR Engine
Tesseract is strictly required for the text extraction phase.

#### Windows
~~~
Run the included tesseract-ocr-w64-setup executable.
~~~

##### Note: Make sure your Python script points to the installation path (e.g., pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe').

#### Linux
~~~
sudo apt-get update

sudo apt-get install tesseract-ocr
~~~

#### MacOS
~~~
brew install tesseract
~~~

## 💻 Usage

Run the main program to start the traffic monitoring simulation using the provided sample videos:

~~~
python Traffic_Monitoring_Program.py
~~~

#### Press q (or the mapped exit key in your script) to terminate the video window and end the program.

## 🛣️ Roadmap & Future Scope

~~~
[ ] Database Integration: Log violator data (Plate Number, Timestamp, Speed) into a local SQLite or remote PostgreSQL database.

[ ] Web Interface: Connect the existing HTML components to a Flask/FastAPI backend for a live monitoring dashboard.

[ ] Edge Deployment: Optimize model inference using OpenVINO or TensorRT for deployment on Raspberry Pi or Jetson Nano.

[ ] Night Vision/Low Light: Improve plate detection accuracy in poor lighting conditions using image enhancement techniques.
~~~
