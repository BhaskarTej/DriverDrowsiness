import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
import pygame.mixer as mixer

# Initialize  sound mixer
mixer.init()
# alert sound
alert_sound = mixer.Sound('alert.wav')

# Load cascasdes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'face.xml')
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'left_eye.xml')
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'right_eye.xml')
eye_status_labels = ['Closed', 'Open']
# Load the drowsiness detection model
drowsiness_model = load_model('models/drowsiness_model.h5')
current_dir = os.getcwd()
# Open the video capture device (webcam)
video = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
count = 0
score = 0
thickness = 2

# Main loop 
while True:
    # Read a frame from the video feed
    ret, frame = video.read()
    # Get the height and width of the frame
    height, width = frame.shape[:2]
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # Detect left eyes 
    left_eye = left_eye_cascade.detectMultiScale(gray)
    # Detect right eyes 
    right_eye = right_eye_cascade.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), -1)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 2)

    # Loop through each detected right eye
    for (x, y, w, h) in right_eye:
        roi = gray[y:y+h, x:x+w]
       
        count += 1
        eye_img = cv2.resize(roi, (24, 24))
        eye_img = eye_img / 255.0
        eye_img = eye_img.reshape(24, 24, -1)
        eye_img = np.expand_dims(eye_img, axis=0)
        pred = drowsiness_model.predict_classes(eye_img)
        if pred[0] == 1:
            eye_status_labels = 'Open'
        elif pred[0] == 0:
            eye_status_labels = 'Closed'
        # Exit the loop after processing the first detected right eye
        break

    # Loop through each detected left eye
    for (x, y, w, h) in left_eye:
        roi = gray[y:y+h, x:x+w]
        count += 1
        eye_img = cv2.resize(roi, (24, 24))
        eye_img = eye_img / 255.0
        eye_img = eye_img.reshape(24, 24, -1)
        eye_img = np.expand_dims(eye_img, axis=0)
        pred = drowsiness_model.predict_classes(eye_img)
        if pred[0] == 1:
            eye_status_labels = 'Open'
        elif pred[0] == 0:
            eye_status_labels = 'Closed'
        # Exit the loop after processing the first detected left eye
        break

    # Update the score based on the prediction
    if pred[0] == 0:
        score += 1
    else:
        score -= 1

    # Ensure the score is non-negative
    score = max(0, score)

    # Display the eye status and score on the frame
    status_text = 'Closed' if pred[0] == 0 else 'Open'
    cv2.putText(frame, status_text, (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Score: ' + str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 15:
        # Save a snapshot of the frame
        cv2.imwrite(os.path.join(current_dir, 'alert.jpg'), frame)
        try:
            # Play the alert sound
            alert_sound.play()
        except:
            pass
        thickness = min(16, thickness + 2)
        thickness = max(2, thickness - 2)
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thickness)

    # Display the frame
    cv2.imshow('Drowsiness Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
video.release()
cv2.destroyAllWindows()
