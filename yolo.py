from ultralytics import YOLO
import cv2
import streamlit as st

st.title("YOLO11 AI Object Detection")
st.header("Ai bilan obyektlarni tanib va sanay olish")

# Kamera ochish
cap = cv2.VideoCapture(0)

# Modelni chaqirish
model = YOLO("yolo11n.pt")  # Nano model tezroq

# Streamlit uchun konteyner
frame_placeholder = st.empty()


def main():
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Kamera ishlamayapti")
            break

        # Frame o'lchamini sozlash
        frame = cv2.resize(frame, (640, 480))

        # YOLO bilan obyekt aniqlash
        result = model(frame, verbose=False)

        # Annotatsiya qilingan frame
        annotated_frame = result[0].plot()

        # Obyektlarni sanash
        boxes = result[0].boxes
        person_count = sum(1 for box in boxes if model.names[int(box.cls[0])] == "person")

        # FPS va odam sonini frame ustiga yozish
        cv2.putText(annotated_frame, f"Odamlar: {person_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # OpenCV BGR → RGB (Streamlit uchun)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Streamlit orqali ko‘rsatish
        frame_placeholder.image(annotated_frame, channels="RGB")

        # ESC yoki 'q' tugmasini tekshirish
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
