import face_recognition
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

path = "data"

def getImageID(path):
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    emotion_labels = []
    ids = []

    for imagePaths in imagePath:
        faceImage = face_recognition.load_image_file(imagePaths)
        faceLocations = face_recognition.face_locations(faceImage)

        for faceLocation in faceLocations:
            top, right, bottom, left = faceLocation
            faceImageAligned = face_recognition.face_encodings(faceImage, [faceLocation], num_jitters=10)[0]

            emotion_label = os.path.split(imagePaths)[-1].split(".")[3]
            Id = int(os.path.split(imagePaths)[-1].split(".")[1])

            faces.append(faceImageAligned)
            emotion_labels.append(emotion_label)
            ids.append(Id)

    return ids, faces, emotion_labels

IDs, facedata, emotion_labels = getImageID(path)

# Show progress bar for training
for i in tqdm(range(100), desc='Training'):
    # Perform training steps here
    # ...

# Save the trained model
 np.savez("Trainer.npz", facedata=facedata, IDs=IDs, emotion_labels=emotion_labels)
 plt.close('all')
 print("Đã xử lý dữ liệu thành công!")
