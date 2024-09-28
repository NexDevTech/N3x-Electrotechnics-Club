import cv2
import mediapipe as mp
import math
#Media pipe imstall script:
#pip install mediapipe

#Camera select
cam_select = 0

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(cam_select)

#Source Control Message-Commit Test 

cap = cv2.VideoCapture(cam_select)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

  while cap.isOpened():
    success, image = cap.read()
    image_height, image_width, _ = image.shape
    #to improve performance
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        x_max = 0
        y_max = 0
        x_min = image_width
        y_min = image_height
        for ids, landmrk in enumerate(hand_landmarks.landmark):
            cx = int(landmrk.x * image_width) # position x
            cy = int(landmrk.y * image_height) # position Y
            if cx > x_max:
                x_max = cx
            if cx < x_min:
                x_min = cx
            if cy > y_max:
                y_max = cy
            if cy < y_min:
                y_min = cy
############################################################################################################################################################################
#QUEST:
            #0 -create  variables/list/field for every landmarks (0-20)  (handmark_number=ids,position x, position y)
            #1 -calculate distance between two points (look math "sqrt" and "hypot")
            #2 -recognize fist (wrist_position and position of EVERY finger TIP at some distance away from wrist) 
            #3 -calculate how many fingers is opened....and print it, finger tip is at some distance away from MPC
            #4 -recognize gesto and print its name (compared position of  landmarks and recognize gesto)
            #5 -make it work at any distance away from camera #chek distance from wrist to finger tip and compare it with lenth from wrist to tip
            #6 -slice program for better reading, make separate gesture program


############################################################################################################################################################################
    # Flip the image horizontally for a mirror-view
    cv2.imshow('HandTracking', cv2.flip(image, 1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()