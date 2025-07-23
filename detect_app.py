import torch
import cv2
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set model to eval mode and device
model.eval()

# Open the webcam
cap = cv2.VideoCapture(0)  # Use 0 for default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model(img_rgb)

    # Parse results
    detections = results.xyxy[0]  # (x1, y1, x2, y2, confidence, class)

    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        confidence = float(conf)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Real-Time Object Detection', frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
