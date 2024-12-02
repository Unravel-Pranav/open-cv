import cv2
import mediapipe as mp

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
source = cv2.VideoCapture(0)
pose = mp_pose.Pose()

win_name = "Movement and Sentiment Detection"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

while cv2.waitKey(1) != 27:  # Exit on 'ESC'
    has_frame, frame = source.read()
    if not has_frame:
        break

    frame = cv2.flip(frame, 1)  # Flip frame horizontally
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Pose
    results = pose.process(rgb_frame)

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

        # Example: Detect raised hands for potential sentiment or action
        left_hand_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
        right_hand_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
        nose_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y

        if left_hand_y < nose_y or right_hand_y < nose_y:
            cv2.putText(frame, "Raised Hands Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow(win_name, frame)

# Release resources
source.release()
cv2.destroyWindow(win_name)