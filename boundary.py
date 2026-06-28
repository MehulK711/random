import cv2
import mediapipe as mp

# Initialize Mediapipe modules
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up boundaries (these can be adjusted)
BOUNDARY_X_MIN, BOUNDARY_X_MAX = 100, 500
BOUNDARY_Y_MIN, BOUNDARY_Y_MAX = 100, 400

def check_boundary(x, y):
    # Function to check if point is outside the boundary
    if x < BOUNDARY_X_MIN or x > BOUNDARY_X_MAX or y < BOUNDARY_Y_MIN or y > BOUNDARY_Y_MAX:
        return False
    return True

cap = cv2.VideoCapture(0)

# Mediapipe models for face and hand detection
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
        mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Skipping empty frame")
            continue

        # Convert the image color to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape

        # Process face detection
        face_results = face_detection.process(image_rgb)

        # Process hand detection
        hand_results = hands.process(image_rgb)

        # Draw boundary box on the image
        cv2.rectangle(image, (BOUNDARY_X_MIN, BOUNDARY_Y_MIN), (BOUNDARY_X_MAX, BOUNDARY_Y_MAX), (0, 255, 0), 2)

        # Check face boundary
        if face_results.detections:
            for detection in face_results.detections:
                # Get the bounding box coordinates for the face
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * image_width)
                y = int(bboxC.ymin * image_height)
                w = int(bboxC.width * image_width)
                h = int(bboxC.height * image_height)

                # Draw face bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Check if face is outside the boundary
                if not check_boundary(x + w // 2, y + h // 2):  # Check center of the face
                    cv2.putText(image, "Face out of boundary!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Check hand boundary
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Extract wrist landmark (landmark 0)
                wrist_x = int(hand_landmarks.landmark[0].x * image_width)
                wrist_y = int(hand_landmarks.landmark[0].y * image_height)

                # Draw hand landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check if hand is outside the boundary
                if not check_boundary(wrist_x, wrist_y):
                    cv2.putText(image, "Hand out of boundary!", (wrist_x, wrist_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the image
        cv2.imshow('Gesture and Face Boundary Detection', image)

        # Exit on pressing 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
