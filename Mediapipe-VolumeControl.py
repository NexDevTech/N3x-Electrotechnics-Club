import cv2
import mediapipe as mp
import math
from ctypes import cast, POINTER
###################################
################!!!################
#!!![pip install pycaw]!!!
################!!!################
###################################
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume 

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# Get system audio controller (pycaw)
devices = AudioUtilities.GetSpeakers() 
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

vol_range = volume.GetVolumeRange()  # Get the volume range (0 to 100%)
min_vol = vol_range[0]
max_vol = vol_range[1]

# Camera Select:
cap = cv2.VideoCapture(0)

# Define the area for the upper-right corner
corner_threshold_width = 0.5  
corner_threshold_height = 0.5  

volume_level = 0  # Initialize volume level
x_text, y_text = 640, 50  # Initialize text position

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        image_height, image_width, _ = image.shape

        # Improve performance 
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Store landmarks positions
                landmarks_positions = []
                for ids, landmark in enumerate(hand_landmarks.landmark):
                    cx = int(landmark.x * image_width)
                    cy = int(landmark.y * image_height)
                    landmarks_positions.append((ids, cx, cy))

                # Check if the hand is in the upper-right corner of the screen
                wrist_x, wrist_y = landmarks_positions[0][1], landmarks_positions[0][2]  # Wrist position

                if wrist_x >= image_width * corner_threshold_width and wrist_y <= image_height * corner_threshold_height:
                    # Get the thumb tip (landmark 4) and index fingertip (landmark 8)
                    x1, y1 = landmarks_positions[4][1], landmarks_positions[4][2]  # Thumb tip
                    x2, y2 = landmarks_positions[8][1], landmarks_positions[8][2]  # Index fingertip

                    # Calculate the distance between thumb and index finger
                    distance = math.hypot(x2 - x1, y2 - y1)

                    # Normalize the distance to control volume level 
                    min_distance = 30  # Minimum fingers distance 
                    max_distance = 160  # Maximum fingers distance 
                    volume_level = max(min((distance - min_distance) / (max_distance - min_distance), 1.0), 0.0)

                    # Map volume_level to system volume range
                    target_volume = volume_level * (max_vol - min_vol) + min_vol
                    volume.SetMasterVolumeLevel(target_volume, None)

                    # Draw a line between the thumb and index finger
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    #text position 
                    x_text = (x1 + x2) // 2 + 80
                    y_text = (y1 + y2) // 2

                    # console volume print
                    print(f"Distance: {distance:.2f}, Volume: {int(volume_level * 100)}%")

        flipped_image = cv2.flip(image, 1)

        #draw the text after the flip (so it isn't mirrored)
        cv2.putText(flipped_image, f'{int(volume_level * 100)}%', (image_width - x_text, y_text),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 128), 2)

        # Show the flipped image with correct text
        cv2.imshow('HandTracking', flipped_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
