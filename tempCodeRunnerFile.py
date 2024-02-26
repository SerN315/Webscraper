import cv2
import face_recognition
import numpy as np
import keras
from tensorflow.keras.models import load_model

# Load the pre-trained face recognition model
known_faces = np.load("Trainer.npz")
facedata = known_faces["facedata"]
IDs = known_faces["IDs"]

# Load the pre-trained emotion detection model
emotion_model = load_model("fer2013_mini_XCEPTION.102-0.66.hdf5", compile=False)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FPS, 60)  # Set the frame rate to 60 fps
name_list = ["", "Nguyen"]

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all face locations in the frame
    face_locations = face_recognition.face_locations(rgb_frame)

    for (top, right, bottom, left) in face_locations:
        # Extract the face encoding from the current frame
        face_encoding = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])[0]

        # Perform face recognition using the known face encodings
        face_distances = face_recognition.face_distance(facedata, face_encoding)
        min_distance_index = np.argmin(face_distances)
        min_distance = face_distances[min_distance_index]

        if min_distance < 0.6:
            name = name_list[IDs[min_distance_index]]
        else:
            name = "Unknown"

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (119, 221, 119), 1)
        cv2.putText(frame, name, (left, top - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (119, 221, 119), 2)

        # Extract the face ROI for emotion detection
        face_roi = frame[top:bottom, left:right]
        face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi_gray = cv2.resize(face_roi_gray, (64, 64))
        face_roi_gray = face_roi_gray.astype("float") / 255.0
        face_roi_gray = np.expand_dims(face_roi_gray, 0)
        face_roi_gray = np.expand_dims(face_roi_gray, -1)

        # Perform emotion detection
        emotion_preds = emotion_model.predict(face_roi_gray)[0]
        emotion_label = emotion_labels[np.argmax(emotion_preds)]

        # Display the predicted emotion
        cv2.putText(frame, emotion_label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (119, 221, 119), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()