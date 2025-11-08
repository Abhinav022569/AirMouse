import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np # Import numpy for standard deviation

mp_solutions = mp.solutions

# --- Constants for our logic ---

# --- 
# --- EDIT THIS ONE LINE ---
# --- 
CAMERA_INDEX = 0  # 0 = Your Laptop Camera, 1 = Your DroidCam (use find_cameras.py to check)
# ---
# ---
# ---

SMOOTHING_FACTOR = 0.5    # How much to smooth. 0.0=no smooth, 1.0=no movement. 0.5-0.8 is good.
CLICK_THRESHOLD = 0.05    # Normalized distance for a pinch (0.05 is 5% of hand width)
ACTIVATION_TIMEOUT = 1.5  # Seconds to wait for the next gesture in the sequence
SENSITIVITY = 2.5         # <<< NEW: Multiplies mouse movement. Higher = faster.

# --- Helper function for distance ---
def get_distance(p1, p2):
    """Calculates the 2D Euclidean distance between two MediaPipe landmarks."""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# --- Helper functions for gesture recognition ---
def is_thumbs_up(hand_landmarks):
    """Checks if the hand is in a 'Thumbs Up' gesture."""
    thumb_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.THUMB_IP] # Inner thumb joint
    
    index_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_PIP] # Middle index joint
    
    middle_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]
    
    # Rule: Thumb is extended (tip is above inner joint)
    # AND all other fingers are closed (tips are below middle joints)
    thumb_extended = thumb_tip.y < thumb_ip.y
    fingers_closed = (index_tip.y > index_pip.y) and (middle_tip.y > middle_pip.y)
    
    return thumb_extended and fingers_closed

def is_palm_splayed(hand_landmarks):
    """Checks if the hand is in a 'Splayed Palm' gesture."""
    index_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_PIP]
    
    middle_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]
    
    ring_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.RING_FINGER_PIP]
    
    pinky_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.PINKY_PIP]
    
    # Rule: All 4 fingers are extended (tips are above middle joints)
    fingers_extended = (index_tip.y < index_pip.y) and \
                        (middle_tip.y < middle_pip.y) and \
                        (ring_tip.y < ring_pip.y) and \
                        (pinky_tip.y < pinky_pip.y)
    
    return fingers_extended

# --- Setup for webcam and models ---
mp_hands = mp_solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp_solutions.drawing_utils
# --- END OF SETUP ---


# --- SIMPLE CAMERA LOGIC ---
print(f"Attempting to connect to camera at Index {CAMERA_INDEX}...")
cam = cv2.VideoCapture(CAMERA_INDEX)

# Try to read one frame and check if it's a real image (not blank)
success, test_frame = cam.read()
if not success or (test_frame is not None and np.std(test_frame) < 10):
    print(f"--- ERROR ---")
    print(f"Failed to open camera at Index {CAMERA_INDEX}.")
    print(f"Try changing CAMERA_INDEX at the top of the script.")
    print(f"If 0 doesn't work, try 1. If 1 doesn't work, try 0.")
    print(f"---")
    cam.release()
    exit() # Quit the program
else:
    print(f"Successfully connected to camera at Index {CAMERA_INDEX}.")
# --- END OF LOGIC ---

screen_width, screen_height = pyautogui.size()

# --- 1. Smoothing & Relative Movement Variables ---
# We initialize the smoother to the *current* mouse position
smooth_x, smooth_y = pyautogui.position() 
prev_palm_x, prev_palm_y = 0, 0
first_active_frame = True

# --- 2. State Machine Variables ---
is_active = False           # Is the Air Mouse on or off?
current_state = "IDLE"      # What is the FSM state? "IDLE" or "AWAITING_PALM"
last_gesture_time = 0       # When did we last see a "Thumbs Up"?

# --- 3. Click Lock Variable ---
click_lock = False

print("Air Mouse v2 Running. Show 'Thumbs Up' then 'Palm' to activate/deactivate.")
print("Press 'q' to quit.")

while True:
    success, frame = cam.read()
    if not success:
        continue
        
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # --- Gesture Recognition Logic ---
        thumb_is_up = is_thumbs_up(hand_landmarks)
        palm_is_splayed = is_palm_splayed(hand_landmarks)

        # --- State Machine (Feature 2: Activation) ---
        
        if current_state == "IDLE":
            if thumb_is_up:
                print("STATE CHANGE: Saw Thumbs Up. Awaiting Palm...")
                current_state = "AWAITING_PALM"
                last_gesture_time = time.time()
                
        elif current_state == "AWAITING_PALM":
            # Check for timeout
            if time.time() - last_gesture_time > ACTIVATION_TIMEOUT:
                print("STATE CHANGE: Timeout. Resetting to IDLE.")
                current_state = "IDLE"
            # Check for success
            elif palm_is_splayed:
                is_active = not is_active # Toggle the mouse
                status_str = "ACTIVE" if is_active else "INACTIVE"
                print(f"STATE CHANGE: Saw Palm! Mouse is now {status_str}")
                
                # --- NEW: Reset for Relative Mode ---
                if is_active:
                    first_active_frame = True # Prime the relative mode
                    smooth_x, smooth_y = pyautogui.position() # Start from current mouse pos
                # ---
                
                current_state = "IDLE"
                
        # --- End of State Machine ---


        # --- Mouse Control (Only if 'is_active' is True) ---
        if is_active:
            
            # --- Feature 1: Relative Movement Logic ---
            palm_center = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.MIDDLE_FINGER_MCP] # Landmark #9
            
            # On the first frame, just "prime" the previous position
            if first_active_frame:
                prev_palm_x, prev_palm_y = palm_center.x, palm_center.y
                first_active_frame = False
            else:
                # 1. Calculate the change (delta) in hand position
                delta_x = (palm_center.x - prev_palm_x) * screen_width * SENSITIVITY
                delta_y = (palm_center.y - prev_palm_y) * screen_height * SENSITIVITY
                
                # 2. Define the new target by adding the delta to the last smoothed position
                target_x = smooth_x + delta_x
                target_y = smooth_y + delta_y
                
                # 3. Apply smoothing
                smooth_x = (smooth_x * SMOOTHING_FACTOR) + (target_x * (1 - SMOOTHING_FACTOR))
                smooth_y = (smooth_y * SMOOTHING_FACTOR) + (target_y * (1 - SMOOTHING_FACTOR))
                
                # 4. Move the mouse
                pyautogui.moveTo(smooth_x, smooth_y)
                
                # 5. Save the current position for the next frame
                prev_palm_x, prev_palm_y = palm_center.x, palm_center.y

            
            # --- Feature 3: Pinch-to-Click ---
            thumb_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            
            wrist = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.WRIST]
            middle_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]
            hand_size = get_distance(wrist, middle_pip)
            
            if hand_size > 0.01:
                pinch_distance = get_distance(index_tip, thumb_tip) / hand_size
                
                if pinch_distance < CLICK_THRESHOLD:
                    if not click_lock:
                        print("CLICK!")
                        pyautogui.click()
                        click_lock = True
                
                if pinch_distance > (CLICK_THRESHOLD * 1.5):
                    click_lock = False
            
            
        # --- Drawing and Display (Always on) ---
        
        if is_active:
            status_text = "Mouse: ACTIVE"
            color = (0, 255, 0) # Green
        else:
            status_text = "Mouse: INACTIVE"
            color = (0, 0, 255) # Red
            
        state_text = f"State: {current_state}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, state_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    cv2.imshow("Air Mouse v2", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cam.release()
cv2.destroyAllWindows()