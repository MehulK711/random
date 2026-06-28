import cv2
import mediapipe as mp
import pyautogui
import screen_brightness_control as sbc

def gesture_control():
    # Initialize Video Capture
    cap = cv2.VideoCapture(0)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    drawing_utils = mp.solutions.drawing_utils

    while True:
        ret, image = cap.read()
        if not ret:
            break
        
        frame_height, frame_width, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image to detect hands
        output = hands.process(rgb_image)
        hand_landmarks = output.multi_hand_landmarks
        hand_labels = output.multi_handedness

        if hand_landmarks:
            for hand_landmark, hand_label in zip(hand_landmarks, hand_labels):
                # Draw hand landmarks on the image
                drawing_utils.draw_landmarks(image, hand_landmark, mp_hands.HAND_CONNECTIONS)
                
                landmarks = hand_landmark.landmark
                thumb_tip_x, thumb_tip_y = None, None
                pinky_tip_x, pinky_tip_y = None, None
                index_tip_x, index_tip_y = None, None
                
                for id, landmark in enumerate(landmarks):
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)
                    
                    # Thumb tip (id = 4)
                    if id == 4:
                        cv2.circle(image, (x, y), 8, (0, 0, 255), 3)
                        thumb_tip_x, thumb_tip_y = x, y
                    
                    # Pinky tip (id = 20)
                    if id == 20:
                        cv2.circle(image, (x, y), 8, (255, 0, 255), 3)
                        pinky_tip_x, pinky_tip_y = x, y

                    # Index finger tip (id = 8)
                    if id == 8:
                        cv2.circle(image, (x, y), 8, (0, 255, 255), 3)
                        index_tip_x, index_tip_y = x, y
                
                # Determine hand label (left or right)
                hand_type = hand_label.classification[0].label
                print(f"Detected hand: {hand_type}")

                # Volume control using the right hand
                if hand_type == 'Right':
                    if thumb_tip_x is not None and pinky_tip_x is not None:
                        dist_volume = int(((pinky_tip_x - thumb_tip_x) ** 2 + (pinky_tip_y - thumb_tip_y) ** 2) ** 0.5)
                        
                        # Draw a line between the thumb and pinky
                        cv2.line(image, (thumb_tip_x, thumb_tip_y), (pinky_tip_x, pinky_tip_y), (0, 200, 0), 5)
                        
                        # Control volume based on distance
                        if dist_volume > 100:
                            pyautogui.press('volumeup')
                            print("Volume UP")
                        elif dist_volume < 50:
                            pyautogui.press('volumedown')
                            print("Volume DOWN")
                
                # Brightness control using the left hand
                if hand_type == 'Left':
                    if index_tip_x is not None and thumb_tip_x is not None:
                        dist_brightness = int(((thumb_tip_x - index_tip_x) ** 2 + (thumb_tip_y - index_tip_y) ** 2) ** 0.5)
                        
                        # Draw a line between the thumb and index finger
                        cv2.line(image, (index_tip_x, index_tip_y), (thumb_tip_x, thumb_tip_y), (200, 0, 0), 5)
                        
                        # Adjust brightness based on distance
                        if dist_brightness > 50:
                            sbc.set_brightness('+10')
                            print("Brightness UP")
                        else:
                            sbc.set_brightness('-10')
                            print("Brightness DOWN")
        
        # Show the image with hand landmarks and the lines
        cv2.imshow("Gesture Control", image)
        
        # Break the loop if 'e' is pressed
        if cv2.waitKey(10) & 0xFF == ord('e'):
            break
    
    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    gesture_control()
