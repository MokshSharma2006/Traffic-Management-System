import cv2
import numpy as np
import pytesseract
import easyocr
import threading
import tkinter as tk
import webbrowser
import os
import time
from tkinter import Label, Button, Frame
from flask import Flask, render_template_string
from ultralytics import YOLO
from scipy.spatial import distance

# Flask App
app = Flask(__name__)

# Load YOLO models (vehicles & license plates)
import torch

torch.serialization.add_safe_globals(["ultralytics.nn.tasks.DetectionModel"])  # Allowlist detection model
vehicle_model = YOLO("yolov8n.pt")  # Load YOLO

plate_model = YOLO("license_plate_detector.pt")  # License plate detection (Download this)

# Initialize EasyOCR for plate text recognition
reader = easyocr.Reader(['en'])

# Set Tesseract Path (Update if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Open video file
video = cv2.VideoCapture("traffic2.mp4")
video.set(cv2.CAP_PROP_FPS, 100)
fps = video.get(cv2.CAP_PROP_FPS)
real_distance_meters = 10
pixel_distance = 400
pixels_per_meter = pixel_distance / real_distance_meters
speed_limit = 60  # km/h

# Data Storage
vehicle_data = []
vehicle_tracks = {}
vehicle_speeds = {}
vehicle_last_seen = {}
frame_count = 0
running = True

# GUI Setup
gui = tk.Tk()
gui.title("Traffic Monitoring System")
gui.geometry("500x500")
gui.configure(bg="#2C3E50")

header = Label(gui, text="Traffic Monitoring System", font=("Arial", 18, "bold"), fg="white", bg="#2C3E50")
header.pack(pady=20)

frame = Frame(gui, bg="#34495E", bd=5, relief=tk.RIDGE)
frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

# Function to Start Video Processing
def start_processing():
    global running
    running = True
    threading.Thread(target=process_video).start()

# Function to Stop Video Processing
def stop_processing():
    global running
    running = False

# Function to Open Report in Browser
def view_report():
    webbrowser.open("http://127.0.0.1:5000")


def quit_processing():
    global running
    running = False  # Stop video processing
    gui.destroy()  # Properly close the Tkinter window
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Thank You For Using Our Program")
    time.sleep(2)
    video.release()
    cv2.destroyAllWindows()
    quit()


button_style = {"font": ("Arial", 12, "bold"), "width": 20, "height": 2, "bd": 3}

start_button = Button(frame, text="▶ Start Monitoring", command=start_processing, **button_style, bg="#27AE60", fg="white")
start_button.pack(pady=10)

stop_button = Button(frame, text="■ Stop Monitoring", command=stop_processing, **button_style, bg="#0ba1c9", fg="white")
stop_button.pack(pady=10)

report_button = Button(frame, text="📊 View Report", command=view_report, **button_style, bg="#2980B9", fg="white")
report_button.pack(pady=10)

quit_button = Button(frame, text="⚠️ Quit" ,command=quit_processing, **button_style, bg="#E74C3C", fg="white")
quit_button.pack(pady=10)

# Function to Process Video Frames
def process_video():
    global running, frame_count
    while running and video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame_count += 1
        frame = process_frame(frame)

        # Display video
        cv2.imshow("Traffic Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

# Function to Detect Vehicles & Read License Plates
def process_frame(frame):
    global vehicle_tracks, vehicle_speeds, vehicle_last_seen
    results = vehicle_model(frame)
    new_tracks = {}

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            conf = box.conf[0]

            if conf > 0.6 and class_id in [2, 3, 5, 7]:  # Cars, trucks, buses, motorbikes
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                new_tracks[center] = (x1, y1, x2, y2, class_id)

    for center, (x1, y1, x2, y2, class_id) in new_tracks.items():
        closest_id = None
        min_dist = float("inf")

        for track_id, prev_center in vehicle_tracks.items():
            dist = distance.euclidean(center, prev_center)
            if dist < min_dist and dist < 50:
                min_dist = dist
                closest_id = track_id

        if closest_id is None:
            closest_id = len(vehicle_tracks) + 1

        vehicle_tracks[closest_id] = center

        if closest_id in vehicle_last_seen:
            time_elapsed = (frame_count - vehicle_last_seen[closest_id]) / fps
            if time_elapsed > 0:
                speed = (min_dist / pixels_per_meter) / time_elapsed * 3.6
                avg_speed = np.mean(vehicle_speeds.get(closest_id, [speed])[-5:])
                vehicle_speeds.setdefault(closest_id, []).append(avg_speed)
            else:
                avg_speed = 0
        else:
            avg_speed = 0

        vehicle_last_seen[closest_id] = frame_count

        # Detect License Plate
        plate_results = plate_model(frame)
        plate_number = "Unknown"

        for plate in plate_results:
            for pbox in plate.boxes:
                px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                plate_roi = frame[py1:py2, px1:px2]

                gray_plate = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                text = reader.readtext(gray_plate, detail=0)
                plate_number = text[0] if text else "Unknown"

        # Draw bounding boxes
        color = (0, 255, 0) if avg_speed <= speed_limit else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"Speed: {int(avg_speed)} km/h", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Store Data
        vehicle_data.append([class_id, int(avg_speed), plate_number])

    return frame

@app.route('/')
def display_table(): # Jinja Code For Web Page
    return render_template_string("""             
    <!DOCTYPE html>
    <html>
    <head>
        <title>Traffic Monitoring Report</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css">
        <style>
            body { background-color: #f4f6f7; font-family: 'Arial', sans-serif; text-align: center; }
            .container { max-width: 900px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); }
            h2 { color: #2C3E50; margin-bottom: 20px; }
            th { background: #007bff; color: white; }
            table { margin-top: 20px; }
            .footer { margin-top: 30px; font-size: 14px; color: #777; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🚦 Traffic Monitoring Report</h2>
            <table class="table table-striped table-hover">
                <thead class="table-dark">
                    <tr><th>Vehicle Type</th><th>Speed (km/h)</th><th>Plate Number</th></tr>
                </thead>
                <tbody>
                    {% for row in data %}
                    <tr>
                        <td>{{ row[0] }}</td>
                        <td>{% if row[1] > 60 %}<span style="color: red; font-weight: bold;">{{ row[1] }}</span>{% else %}<span style="color: green; font-weight: bold;">{{ row[1] }}</span>{% endif %}</td>
                        <td>{{ row[2] }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            <div class="footer">© 2024 Traffic Monitoring System | All Rights Reserved</div>
        </div>
    </body>
    </html>
    """, data=vehicle_data)

if __name__ == '__main__':
    threading.Thread(target=lambda: app.run(debug=True, port=5000, use_reloader=False)).start()
    gui.mainloop()
    video.release()
    cv2.destroyAllWindows()
