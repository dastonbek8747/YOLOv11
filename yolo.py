from ultralytics import YOLO
import cv2
import streamlit as st

st.title("YOLO11 AI Object Detection")

# Video fayl yoki webcam (localda ishlasa)
video_path = "video.mp4"  # yoki 0 agar local kompyuterda webcam ishlasa
cap = cv2.VideoCapture(video_path)

model = YOLO("yolo11n.pt")  # Nano model tezroq

frame_placeholder = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    result = model(frame, verbose=False)
    annotated_frame = result[0].plot()

    # Obyektlarni sanash
    boxes = result[0].boxes
    person_count = sum(1 for box in boxes if model.names[int(box.cls[0])] == "person")
    cv2.putText(annotated_frame, f"Odamlar: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # BGR -> RGB
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(annotated_frame, channels="RGB")
