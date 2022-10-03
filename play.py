import cv2

cam = cv2.VideoCapture(0)

while True:
    check, frame = cam.read()

    cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == 32: # Spacebar
        img1 = frame
        cv2.imshow("image1", img1)

cam.release()
cv2.destroyAllWindows()
