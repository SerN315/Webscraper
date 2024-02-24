

#testing
import cv2
import face_recognition
import numpy as np

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FPS, 60)  # Set the frame rate to 60 fps
name_list = ["", "Nguyen"]

# Load the pre-trained face recognition model
known_faces = np.load("Trainer.npz")
facedata = known_faces["facedata"]
IDs = known_faces["IDs"]

while True:
    ret, frame = video.read()
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
            name = "khong biet luon"

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (119, 221, 119), 1)
        cv2.putText(frame, name, (left, top - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (119, 221, 119), 2)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
print("Data collection completed!")