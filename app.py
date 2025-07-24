from flask import Flask, render_template, Response, jsonify
import torch
import cv2
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

# Load YOLOv8s model
model = YOLO('yolov8s.pt')

# Initialize camera
camera = cv2.VideoCapture(0)
running = False

# Load MiDaS model for depth estimation
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# Load MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# Distance estimation function
def estimate_distance(frame, ymin, ymax, xmin, xmax):
    try:
        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = midas_transforms(input_image).to(device)

        with torch.no_grad():
            prediction = midas(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=input_image.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        roi = depth_map[int(ymin):int(ymax), int(xmin):int(xmax)]

        if roi.size == 0:
            return None

        mean_depth = np.mean(roi)

        # Real-world distance estimation
        pixel_height = ymax - ymin
        assumed_real_height_m = 1.7  # meters
        focal_length_px = 1.2 * frame.shape[0]
        distance_m = (assumed_real_height_m * focal_length_px) / (pixel_height + 1e-6)

        return round(float(distance_m), 2)

    except Exception as e:
        print("Depth Estimation Error:", e)
        return None

# Frame generator
def generate_frames():
    global running
    while running:
        success, frame = camera.read()
        if not success:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start():
    global running
    running = True
    return jsonify({'status': 'started'})

@app.route('/stop')
def stop():
    global running
    running = False
    return jsonify({'status': 'stopped'})

@app.route('/detections')
def detections():
    if not camera.isOpened():
        return jsonify([])

    ret, frame = camera.read()
    if not ret:
        return jsonify([])

    results = model(frame)
    dets = results[0].boxes

    detections = []
    for box in dets:
        xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        name = model.names[cls]

        # Estimate distance using MiDaS
        distance = estimate_distance(frame, ymin, ymax, xmin, xmax)

        detection = {
            "name": name,
            "conf": conf,
            "xmin": int(xmin),
            "ymin": int(ymin),
            "distance_m": distance if distance is not None else "N/A"
        }
        detections.append(detection)

    return jsonify(detections)

if __name__ == '__main__':
    app.run(debug=False)
