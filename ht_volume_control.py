import cv2
import time
import numpy as np
import hand_tracking_module as htm
import math

#Volume manip modules
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#Init volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

#Get and set volume range min and max
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]


#Hand tracking module
hand_detector = htm.handDetector(detection_confidence=0.7)

#Declare width+height of capture frame
w_cam, h_cam = 640, 480

#Capture webcam
cap = cv2.VideoCapture(0)

#Frame capture size
cap.set(3, w_cam) #3 == width
cap.set(4, h_cam) #4 == height

#For fps calculation
p_time = 0
c_time = 0

while True:

    #Read capture 
    success, img = cap.read()

    #For vol display
    vol_percent = np.interp(volume.GetMasterVolumeLevel(), [-65,0], [0,100])
    vol_percent_text = f"Vol: {int(vol_percent)}"

    #Find and draw landmarks
    hand_detector.find_hands(img)
    lm_list = hand_detector.findPosition(img, draw=False)


    if len(lm_list) != 0:

        #End points of thumb and index finger
        x1 = lm_list[4][1] #[4] == point thumb (refer to hand index)
        y1 = lm_list[4][2] #[2] == y co.ord of thumb (refer to print lm_list)

        x2 = lm_list[8][1] #[4] == point index finger (refer to hand index)
        y2 = lm_list[8][2] #[2] == y co.ord of index (refer to print lm_list)

        #Center co.ord between thumb and index landmarks
        cx,cy = (x1+x2)//2, (y1+y2)//2

        #Thumb and index finger circles
        cv2.circle(img, (x1,y1), 15, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 15, (255,0,255), cv2.FILLED)

        #Circle inbetween thumb and index
        cv2.circle(img, (cx,cy), 15, (255,255,255), cv2.LINE_4)
        cv2.circle(img, (cx,cy), 8, (255,255,255), cv2.FILLED)

        #Line between two circles
        cv2.line(img, (x1,y1), (x2,y2), (255,0,255), 3)

        #Hypotenuse of thumb and index
        #This is the length between the thumb and index
        length = math.hypot(x2-x1,y2-y1)

        #Hand range is from 25 to 250
        #We need to map to -65 to 0 (our volume range)
        vol = np.interp(length, [25,200],[min_vol, max_vol])
        
        #Only change vol if "v" is pressed
        if cv2.waitKey(1) & 0xFF == ord('v'):
            volume.SetMasterVolumeLevel(vol, None)
        
        #Draw circle if length between thumb and index less than 50
        if length<25:
            cv2.circle(img, (cx,cy), 15, (255,255,255), cv2.FILLED)
            

    #For fps calculation
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    fps_text = f"FPS: {int(fps)}"
    p_time = c_time


    #Dispay fps on screen
    cv2.putText(img, fps_text,(10,70), cv2.FONT_HERSHEY_PLAIN, 3,
    (255,0,255), 3)

    #Disaply volume percent
    cv2.putText(img, vol_percent_text,(10,140), cv2.FONT_HERSHEY_PLAIN, 3,
    (0,0,0), 3)

    #Display visuals
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)