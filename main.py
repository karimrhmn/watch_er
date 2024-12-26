# Import Libraries
import cv2
import mediapipe as mp

# Mediapipe setup #

# Grabbing the Holistic Model from Mediapipe and
# Initializing the Model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic( 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils


# Main setup #

# Init the video camera
cap = cv2.VideoCapture(0)


while cap.isOpened():

    # Read frame by frame
    ret, frame = cap.read()

    # Resize frame 
    frame = cv2.resize(frame, (800, 800))

    # Converting the from BGR to RGB then back to BGR
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
    # Making predictions using holistic model
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Drawing the Facial Landmarks
    mp_drawing.draw_landmarks(
      image, results.face_landmarks,
      mp_holistic.FACEMESH_CONTOURS,
      mp_drawing.DrawingSpec(
        color=(255, 0, 255),
        thickness=1,
        circle_radius=1
      ),
      mp_drawing.DrawingSpec(
        color=(0, 255, 255),
        thickness=1,
        circle_radius=1
      )
    )

     # Drawing Right hand Land Marks
    right = mp_drawing.draw_landmarks(
      image, 
      results.right_hand_landmarks, 
      mp_holistic.HAND_CONNECTIONS
    )
 
    # Drawing Left hand Land Marks
    left = mp_drawing.draw_landmarks(
      image, 
      results.left_hand_landmarks, 
      mp_holistic.HAND_CONNECTIONS
    )

              

    # Display the resulting image
    cv2.imshow(" Landmarks", image)
 
    # Enter key 'q' to break the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# When all the process is done
# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
