import cv2
import mediapipe as mp
import math

# Declaration for better writing
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Camera Select:
cap = cv2.VideoCapture(2)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        image_height, image_width, _ = image.shape

        # To improve performance
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert the BGR image to RGB
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert the RGB image back to BGR

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Show hand landmarks on screen
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                landmarks_positions = []

                # Hand orientation points(Landmarks) placement
                for ids, landmark in enumerate(hand_landmarks.landmark):
                    cx = int(landmark.x * image_width)  # position x
                    cy = int(landmark.y * image_height)  # position y
                    landmarks_positions.append((ids, cx, cy))  # Append tuple (id, x, y) to list


                # Calculate the distance between two points (quest 1)  
                if len(landmarks_positions) >= 9:
                    x1, y1 = landmarks_positions[4][1], landmarks_positions[4][2]  # Thumb tip
                    x2, y2 = landmarks_positions[8][1], landmarks_positions[8][2]  # Index fingertip

                    # Calculate distance 
                    distance = math.hypot(x2 - x1, y2 - y1)

                    # Print the calculated distance between thumb tip and index fingertip
                    print(f"Distance between thumb tip and index fingertip: {distance:.2f}")

        # Flip the image horizontally for a mirror-view display
        cv2.imshow('HandTracking', cv2.flip(image, 1))

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

cap.release()
cv2.destroyAllWindows()
