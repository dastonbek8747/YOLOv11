import cv2

# img = cv2.imread('img.png')
# cv2.imshow("Rasm", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow('Kamera', frame)
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

