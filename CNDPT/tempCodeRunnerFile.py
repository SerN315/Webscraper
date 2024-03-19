
import cv2
import face_recognition
import numpy as np
from keras.models import load_model
import threading

# Load the pre-trained face recognition model
known_faces = np.load("Trainer.npz")
facedata = known_faces["facedata"]
IDs = known_faces["IDs"]

# Load the pre-trained emotion detection model
emotion_model = load_model("fer2013_mini_XCEPTION.102-0.66.hdf5", compile=False)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FPS, 30)  # Set the frame rate to 30 fps

name_list = ["", "Nguyen", "Khanh"]

# Initialize mutex lock for synchronization
mutex = threading.Lock()

# Initialize variables for face recognition and emotion detection results
face_name = ""
emotion_label = ""

# Initialize an empty frame for display
display_frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Function for face recognition
def recognize_face(frame):
    global face_name

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    for (top, right, bottom, left) in face_locations:
        face_encoding = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])[0]

        face_distances = face_recognition.face_distance(facedata, face_encoding)
        min_distance = min(face_distances)
        min_distance_index = np.argmin(face_distances)

        if min_distance < 0.45:
            face_name = name_list[IDs[min_distance_index]]
        else:
            face_name = "Unknown"

# Function for emotion detection
def detect_emotion(frame, top, right, bottom, left):
    global emotion_label

    face_roi = frame[top:bottom, left:right]
    face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    face_roi_gray = cv2.resize(face_roi_gray, (64, 64))
    face_roi_gray = face_roi_gray.astype("float") / 255.0
    face_roi_gray = np.expand_dims(face_roi_gray, 0)
    face_roi_gray = np.expand_dims(face_roi_gray, -1)

    emotion_preds = emotion_model.predict(face_roi_gray)[0]
    emotion_label = emotion_labels[np.argmax(emotion_preds)]

# Thread function for processing frames
def process_frames():
    global face_name, emotion_label, display_frame

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Clone the frame for independent processing
        cloned_frame = frame.copy()

        # Convert the frame to RGB format
        rgb_frame = cv2.cvtColor(cloned_frame, cv2.COLOR_BGR2RGB)

        # Find all face locations in the frame
        face_locations = face_recognition.face_locations(rgb_frame)

        # Process each face in the frame
        for (top, right, bottom, left) in face_locations:
            # Perform face recognition
            recognize_face(cloned_frame)

            # Perform emotion detection
            detect_emotion(cloned_frame, top, right, bottom, left)

            # Draw a rectangle around the face and display name and emotion
            cv2.rectangle(cloned_frame, (left, top), (right, bottom), (119, 221, 119), 1)
            cv2.putText(cloned_frame, face_name, (left, top - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (119, 221, 119), 2)
            cv2.putText(cloned_frame, emotion_label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (119, 221, 119), 2)

        # Acquire the lock to update the frame
        mutex.acquire()
        display_frame = cloned_frame.copy()
        mutex.release()

    video.release()

# Create and start the thread for processing frames
frame_thread = threading.Thread(target=process_frames)
frame_thread.start()

# Main thread for displaying frames
while True:
    # Resize the frame for smoother display
    resized_frame = cv2.resize(display_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    # Display the frame
    cv2.imshow("Video Feed", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
       break

# Wait for the frame processing thread to finish
frame_thread.join()

cv2.destroyAllWindows()