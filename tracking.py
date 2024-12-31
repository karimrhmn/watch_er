import cv2
import hand_tracking_module as htm
import face_tracking_module as ftm
import time

#For fps calculation
p_time = 0
c_time = 0

#Capture from webcam(0)
cap = cv2.VideoCapture(0)

hand_detector = htm.handDetector()
face_detector = ftm.faceDetector()


while True:

    #Read the frame
    success, img = cap.read()

    #Find and draw hands
    img = hand_detector.find_hands(img, draw=True)

    #Find and draw face
    img = face_detector.find_face(img, draw=True)

    #For fps calculation
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    #Dispay fps on screen
    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_SIMPLEX, 3,
    (255,0,255), 3)
    
    cv2.imshow("Video", img)
    cv2.waitKey(1)