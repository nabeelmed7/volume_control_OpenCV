import cv2
import mediapipe as mp
import time
import hand_tracking_module as hndm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

cap = cv2.VideoCapture(0)
time_previous = 0
time_current = 0
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
detector = hndm.hand_tracking(confidence_detection=0.75, confidence_tracking=0.75)

vol_min = volRange[0]
vol_max = volRange[1]

while True:
    ret, frame = cap.read()
    frame = detector.hands_tracking(frame)
    lmlist = detector.position_find(frame, draw = False)
    if len(lmlist) != 0:
            # lmlist[4], lmlist[8] getting the values for thumb and index finger

            x1, y1 = lmlist[4][1], lmlist[4][2] # co-ordinates of 4 (thumb), 0th element is the id number
            x2, y2 = lmlist[8][1], lmlist[8][2] # co-ordinates of 8 (index finger), 0th element is the id number
            cx, cy = (x1+x2)//2, (y1+y2)//2 # center point of the line
            # circles around thumb and index finger
            cv2.circle(frame, (x1, y1), 17, (0, 0, 255))
            cv2.circle(frame, (x2, y2), 17, (0, 0, 255))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0 ,255), 3) # line to connect them

            length = math.hypot(x2-x1, y2-y1)

            # length range is 30 to 270 and volume range is -96 to 0
            vol = np.interp(length, [30, 270], [vol_min, vol_max])
            volume.SetMasterVolumeLevel(vol, None)

            inner_rect_height = int(np.interp(length, [30, 270], [400, 150]))
            inner_rect_color = (0, 255, 0) if inner_rect_height > 0 else (0, 0, 255)
            cv2.rectangle(frame, (50, inner_rect_height), (85, 400), (0, 255, 0), cv2.FILLED)

            cv2.rectangle(frame, (50, 150), (85, 400), (0, 0, 255), 3)
            

        

    time_current = time.time()
    fps = 1 / (time_current - time_previous)
    time_previous = time_current

    fps_text = "FPS: {:.1f}".format(fps)
    cv2.putText(frame, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    tracking_text = "Volume Control"
    cv2.putText(frame, tracking_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()