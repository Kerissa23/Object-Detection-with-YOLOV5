import streamlit as st
import torch
import cv2
from PIL import Image
import numpy as np

# Set page config FIRST
st.set_page_config(page_title="Real-Time Object Detection", layout="centered")

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

model = load_model()
model.eval()

st.title("📦 Real-Time Object Detection with YOLOv5")
st.markdown("Click **Start Detection** to launch webcam and run object detection in real time.")

start_btn = st.button("Start Detection")
stop_btn = st.button("Stop")

frame_placeholder = st.empty()
cap = None

if start_btn:
    cap = cv2.VideoCapture(0)
    st.success("Webcam started! Press 'Stop' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from webcam.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb)

        for *xyxy, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(rgb, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        frame_placeholder.image(rgb, channels="RGB")

        if stop_btn:
            break

    cap.release()
    st.info("Webcam stopped.")

