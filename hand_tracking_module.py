import cv2
import mediapipe as mp
import time
class hand_tracking():

    def __init__(self, mode=False, model_complexity = 1, max_n_hands=4, confidence_detection=0.5, confidence_tracking=0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.max_n_hands = max_n_hands
        self.confidence_detection = confidence_detection
        self.confidence_tracking = confidence_tracking
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_n_hands, self.model_complexity, self.confidence_detection, self.confidence_tracking)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def hands_tracking(self, frame, draw=True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS, landmark_drawing_spec=None, connection_drawing_spec=self.mpDraw.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1))
        return frame
    
    def position_find(self, frame, handNo=0, draw=True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)
        self.lmandmks_list = []
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmandmks_list.append([id, cx, cy])
                    if draw:
                        cv2.circle(frame, (cx, cy), 15, (0, 255, 255), cv2.FILLED)
        return self.lmandmks_list
    
    def finger_up(self):
        fingers = []
        
        if len(self.lmandmks_list) != 0:
            
            # for thumb
            if self.lmandmks_list[self.tipIds[0]][1] > self.lmandmks_list[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            # llist[8][2] < llist[6][2] means if the y co-ordinate of 8th element(top portion ofindex finger) is higher than the y co=ordinate of 6th element(middle portion of index finger) it means it's open. In open cv0 is highest hence higher is <
            for id in range(1, 5):
                if self.lmandmks_list[self.tipIds[id]][2] < self.lmandmks_list[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers    
    
def main():
    cap = cv2.VideoCapture(0)
    detector = hand_tracking()

    time_previous = 0
    time_current = 0

    while True:
        ret, frame = cap.read()
        frame = detector.hands_tracking(frame)
        lmandmks_list = detector.position_find(frame)
        if len(lmandmks_list) != 0:
            print(lmandmks_list[4])
        time_current = time.time()
        fps = 1 / (time_current - time_previous)
        time_previous = time_current
        fps_text = "FPS: {:.1f}".format(fps)
        cv2.putText(frame, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        tracking_text = "Hand Tracking"
        cv2.putText(frame, tracking_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()