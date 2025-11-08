import cv2
import mediapipe as mp
import pyautogui

# Initialize everything
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1) # We only want to track one hand
mp_draw = mp.solutions.drawing_utils

cam = cv2.VideoCapture(0)

# Get your screen size
screen_width, screen_height = pyautogui.size()

while True:
    success, frame = cam.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Get the landmarks for the first (and only) hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # --- This is the new, important logic ---

        # 1. Get the landmark for the Index Finger Tip (Landmark #8)
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # 2. Map the hand's (x, y) coordinates to your screen's (width, height)
        # The landmark x and y are (0.0 to 1.0), so we multiply by the screen size
        target_x = int(index_finger_tip.x * screen_width)
        target_y = int(index_finger_tip.y * screen_height)

        # 3. Move the mouse!
        pyautogui.moveTo(target_x, target_y)

        # --- End of new logic ---

        # Draw the hand on the preview window (optional, but good for testing)
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Air Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()