import cv2 as cv
import numpy as np
import handtracking as ht
import os
import time

brushThickness = 15
drawColor = (0, 0, 255)  # Fixed red color

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = ht.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

clear_gesture_start_time = 0
CLEAR_GESTURE_DURATION = 2  # Seconds to hold the clear gesture

def clear_canvas():
    global imgCanvas
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)

def save_canvas():
    global imgCanvas
    # Convert to grayscale and apply thresholding to clean up the image
    gray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
    cv.imwrite("saved_canvas.jpg", binary)
    print("Canvas saved")

def is_palm_open(lmList):
    if len(lmList) < 21:
        return False
    finger_tips = [lmList[8][1:], lmList[12][1:], lmList[16][1:], lmList[20][1:]]
    palm_center = lmList[0][1:]
    threshold = 50
    return all(((tip[0] - palm_center[0])**2 + (tip[1] - palm_center[1])**2)**0.5 > threshold for tip in finger_tips)

def is_pinching(lmList):
    if len(lmList) < 21:
        return False
    thumb_tip = lmList[4][1:]
    index_tip = lmList[8][1:]
    distance = ((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)**0.5
    return distance < 50  # Reduced threshold for more sensitive pinching

def gen_frames():
    global xp, yp, imgCanvas, clear_gesture_start_time
    drawing_mode = False

    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            # Flipped horizontally to match natural drawing perspective
            img = cv.flip(img, 1)
            
            # Detect hands
            img = detector.findHands(img)
            lmList = detector.findPosition(img, draw=False)

            if len(lmList) != 0:
                # Get index and thumb tip coordinates
                x1, y1 = lmList[8][1:]
                x2, y2 = lmList[4][1:]

                # Pinching detection for drawing
                if is_pinching(lmList):
                    if not drawing_mode:
                        xp, yp = x1, y1
                        drawing_mode = True
                    
                    # Draw on both preview image and canvas
                    cv.circle(img, (x1, y1), 15, drawColor, cv.FILLED)
                    cv.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                    
                    xp, yp = x1, y1
                else:
                    drawing_mode = False

                # Clear canvas gesture (both palms open)
                if len(detector.results.multi_hand_landmarks) == 2:
                    lmList_left = detector.findPosition(img, handNo=0, draw=False)
                    lmList_right = detector.findPosition(img, handNo=1, draw=False)
                    if is_palm_open(lmList_left) and is_palm_open(lmList_right):
                        if clear_gesture_start_time == 0:
                            clear_gesture_start_time = time.time()
                        elif time.time() - clear_gesture_start_time > CLEAR_GESTURE_DURATION:
                            clear_canvas()
                            print("Canvas cleared")
                            clear_gesture_start_time = 0
                    else:
                        clear_gesture_start_time = 0
                else:
                    clear_gesture_start_time = 0

            # Merge canvas and image
            img_with_canvas = cv.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

            # Add debug information
            cv.putText(img_with_canvas, "Draw: Pinch thumb and index", (10, 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(img_with_canvas, "Clear: Show both palms for 2 seconds", (10, 70), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Encode and yield the frame
            ret, buffer = cv.imencode(".jpg", img_with_canvas)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")