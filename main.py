import cv2
from utils import get_parking_spots_bboxes, empty_or_not
import numpy as np
import time
prev_time = time.time()

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

mask_path = "mask_1929_1080.png"
video_path = "parking_1920_1080_loop.mp4"

mask = cv2.imread(mask_path, 0)
cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

step = 30
spot_status = [False for _ in spots]
frame_nmr = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_nmr % step == 0:
        for i, (x, y, w, h) in enumerate(spots):
            crop = frame[y:y+h, x:x+w]
            
    # 🔍 Detection every N frames
    if frame_nmr % step == 0:
        for i, (x, y, w, h) in enumerate(spots):
            crop = frame[y:y+h, x:x+w]
            spot_status[i] = empty_or_not(crop)

    # 🧮 Counting (REAL-TIME)
    empty_count = sum(1 for s in spot_status if s)
    occupied_count = len(spot_status) - empty_count

   
    # 🎨 Draw boxes
    for i, (x, y, w, h) in enumerate(spots):
        color = (0, 255, 0) if spot_status[i] else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )


    # 📝 Overlay counts
    cv2.putText(
        frame,
        f"Empty: {empty_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Occupied: {occupied_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Total: {len(spots)}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow("Parking Slot Detection", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()