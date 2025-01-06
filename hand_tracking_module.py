import cv2 
import mediapipe as mp
import time


class handDetector():
    def __init__(self,
               image_mode=False,
               max_hands=2,
               detection_confidence=0.5,
               tracking_confidence=0.5):
        
        self.image_mode = image_mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        #Import the hand soloutions
        self.mp_hands = mp.solutions.hands 
        self.hands = self.mp_hands.Hands(
            static_image_mode = self.image_mode,
            max_num_hands = self.max_hands,
            min_detection_confidence = self.detection_confidence,
            min_tracking_confidence = self.tracking_confidence
        )

        #Import the drawing solutions
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
            #Convert to RGB
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    
            #Process hand landmarks
            self.hand_results = self.hands.process(imgRGB)
    
            #Draw hand landmarks for each hand
            if self.hand_results.multi_hand_landmarks:
                for hand in self.hand_results.multi_hand_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img, hand, self.mp_hands.HAND_CONNECTIONS)
            return img    

    def findPosition(self, img, draw=True):

        lm_list = []

        if self.hand_results.multi_hand_landmarks:
            #Hand landmarks for each hand
            for hand in self.hand_results.multi_hand_landmarks:
            
                for id, lm in enumerate(hand.landmark):
                    #print(id, lm) un-comment to see how lm's are presented
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])

                    cv2.circle(img, (cx, cy), 10, (255,0,255),cv2.FILLED)

        return lm_list


def main():

    #For fps calculation
    p_time = 0
    c_time = 0

    #Capture from webcam(0)
    cap = cv2.VideoCapture(0)

    detector = handDetector()

    while True:

        #Read the frame
        success, img = cap.read()
        img = detector.find_hands(img, draw=True)
        
        #Finding key points using hand solutions
        # and printing it
        lm_list = detector.findPosition(img)
        if len(lm_list) != 0:
            print(lm_list[4]) # Print a hand lm here! 4 == thumb tip
        

        #For fps calculation
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        #Dispay fps on screen
        cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_SIMPLEX, 3,
        (255,0,255), 3)
        
        cv2.imshow("Video", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()