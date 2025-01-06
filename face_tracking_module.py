import cv2
import mediapipe as mp
import time

class faceDetector:
    def __init__(self,
                 image_mode=False,
                 num_faces=1,
                 r_landmarks=False,
                 min_detect_confidence=0.5,
                 min_track_confidence=0.5):
        
        # Initialize parameters
        self.static_image_mode = image_mode
        self.max_num_faces = num_faces
        self.refine_landmarks = r_landmarks
        self.min_detection_confidence = min_detect_confidence
        self.min_tracking_confidence = min_track_confidence

        # Import the face mesh solution
        self.mp_face = mp.solutions.face_mesh
        self.face = self.mp_face.FaceMesh(
            static_image_mode=self.static_image_mode,
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

        # Import the drawing utility
        self.mpDraw = mp.solutions.drawing_utils

    def find_face(self, img, draw=True):

        #Convert img to RGB 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Process face landmarks
        self.face_results = self.face.process(imgRGB)

        #Draw hand landmarks for each hand
        if self.face_results.multi_face_landmarks:
            for face in self.face_results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, face, self.mp_face.FACEMESH_CONTOURS)
        return img 


def main():
    #For fps calculation
    p_time = 0
    c_time = 0

    #Capture from webcam(0)
    cap = cv2.VideoCapture(0)

    detector = faceDetector()

    while True:

        #Read the frame
        success, img = cap.read()

        #Find face
        img = detector.find_face(img)
    

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

