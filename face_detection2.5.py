"""
Face Detection using Haarcascade and OpenCV.

This script uses a Haarcascade Classifier to detect faces in real-time using a webcam.
Press 'q' to exit the application.
"""

import pathlib
import cv2
import numpy as np

# Save Haarcascade Frontal Face Cascade Path
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
# Show that path is working from cv2 library
print(cascade_path)

# Load the Classifiers used to detect faces, eyes, nose, etc..
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(str(cascade_path))
eye_cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(str(eye_cascade_path))
smile_cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_smile.xml"
smile_cascade = cv2.CascadeClassifier(str(smile_cascade_path))

# Cat Cascades
cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')


# Since we only have 1 camera
camera = cv2.VideoCapture(0)

# Exit code if the camera cannot be opened and show error
if not camera.isOpened():
    print("Error: Could not open camera")
    exit()

while True:
    # We get a first value that we don't need? and the frame value
    _, frame = camera.read()
    # Get the greyscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    # Parameters:
    # 1.2 - Scale factor for resizing image during detection
    # 5 - Minimum neighbors; higher value results in fewer detections (more confident)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE)

    # Create a blue rectangle around the detected face - BGR COLOR SCHEME for the rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Define ROI
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes within ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(22,22), flags=cv2.CASCADE_SCALE_IMAGE)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Detect smile in the face region
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=10, minSize=(36,36), flags=cv2.CASCADE_SCALE_IMAGE)
        for (sx, sy, sw, sh) in smiles:
            # Draw rectangle around the smile
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 255, 255), 2)

    # Display the current frame with the detected faces highlighted
    cv2.imshow('Face and Eye Detection', frame)

    # Break out of the loop when 'q' is selected
    if cv2.waitKey(1) & 0xFF == ord('q'):  # if you press q, break out of the loop
        break

# Release the camera and close all OPENCV windows
camera.release()
cv2.destroyAllWindows()
