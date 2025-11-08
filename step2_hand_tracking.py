import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils # This will draw the dots and lines

cam = cv2.VideoCapture(0)

while True:
    success, frame = cam.read()

    if success:
        # Flip the frame (so it's like a mirror)
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB (MediaPipe needs RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and find hands
        results = hands.process(frame_rgb)

        # Check if any hands were found
        if results.multi_hand_landmarks:
            # Loop through each hand
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the landmarks (the dots) and connections (the lines)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("My Hand Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()