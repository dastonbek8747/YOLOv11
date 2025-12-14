from ultralytics import YOLO
import cv2
import streamlit as st

st.title("YOLO11 Object Detection")

# Video fayl ishlatamiz (Cloud-da webcam yo'q)
video_path = "video.mp4"  # sizning video faylingiz nomi
cap = cv2.VideoCapture(video_path)

# YOLO modelini yuklash
model = YOLO("yolo11n.pt")  # fayl mavjud bo'lishi kerak

frame_placeholder = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    result = model(frame, verbose=False)
    annotated_frame = result[0].plot()

    # RGB formatga o'tkazish
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(annotated_frame, channels="RGB")

cap.release()
