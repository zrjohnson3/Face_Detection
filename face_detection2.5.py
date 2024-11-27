import pathlib
import cv2
import numpy as np

# Save Haarcascade Frontal Face Cascade Path
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
print(cascade_path)  # Show that path is working from cv2 library

face_cascade = cv2.CascadeClassifier(str(cascade_path))  # Classifier used to detect faces

camera = cv2.VideoCapture(0)  # Since we only have 1 camera

while True:
    _, frame = camera.read()  # We get a first value that we don't need? and the frame value
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Get the greyscale
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2) # BGR COLOR SCHEME

    cv2.imshow('Faces', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # if you press q, break out of the loop
        break

camera.release()
cv2.destroyAllWindows()
