import cv2

# Start your webcam (0 is usually the built-in one)
cam = cv2.VideoCapture(0)

while True:
    # Read an image from the webcam
    success, frame = cam.read()

    if success:
        # Show the image in a window named "My Webcam"
        cv2.imshow("My Webcam", frame)

    # Check if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cam.release()
cv2.destroyAllWindows()