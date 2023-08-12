import cv2
import time
import requests
import numpy as np

DISCORD_WEBHOOK_URL = ''
TEST_MODE = False

if DISCORD_WEBHOOK_URL == '':
    print("Set the DISCORD_WEBHOOK_URL variable to your Discord webhook URL, enabled TEST_MODE for now.")
    TEST_MODE = True

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

prev_frame = None
last_detection_time = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if prev_frame is None:
        prev_frame = gray
        continue
    frame_delta = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.circle(frame, (x + w // 2, y + h // 2), w // 2, (0, 255, 0), 4)
    for contour in contours:
        if cv2.contourArea(contour) < 1:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.circle(frame, (x + w // 2, y + h // 2), max(w, h) // 2, (0, 0, 255), 2)
        current_time = time.time()
        if current_time - last_detection_time >= 0.5:
            current_time = time.time()
            last_detection_time = current_time
            if TEST_MODE:
                print(f"Motion detected, not sending notification because TEST_MODE is enabled")
            else:
                screenshot_1 = np.array(frame)
                screenshot_1_filename = f"screenshot1_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_1_filename, screenshot_1)
                with open(screenshot_1_filename, "rb") as screenshot_1_file:
                    response = requests.post(DISCORD_WEBHOOK_URL, files={"file": screenshot_1_file},
                                             data={"content": f"Motion detected at {time.strftime('%I:%M:%S %p')}"})
                    print("Discord response:", response.text)

    cv2.imshow('Camera Motion/Face Detector', frame)
    prev_frame = gray.copy()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
