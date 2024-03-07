import face_recognition
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import concurrent.futures
import multiprocessing


def process_images(imagePaths):
    faces = []
    ids = []
    emotion_labels = []

    for imagePath in imagePaths:
        faceImage = face_recognition.load_image_file(imagePath)
        faceImage = Image.fromarray(faceImage)  # Convert to PIL Image for resizing
        faceImage = faceImage.resize((256, 256))  # Resize the image to a smaller size
        faceImage = np.array(faceImage)  # Convert back to numpy array
        faceLocations = face_recognition.face_locations(faceImage, model='cnn')

        for faceLocation in faceLocations:
            top, right, bottom, left = faceLocation
            faceImageAligned = face_recognition.face_encodings(faceImage, [faceLocation], num_jitters=5)[0]
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            emotion_label = os.path.split(imagePath)[-1].split(".")[3]

            faces.append(faceImageAligned)
            ids.append(Id)
            emotion_labels.append(emotion_label)

    return ids, faces, emotion_labels


def getImageID(path):
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]
    batch_size = 10  # Number of images to process in each batch
    batches = [imagePath[i:i + batch_size] for i in range(0, len(imagePath), batch_size)]

    faces = []
    ids = []
    emotion_labels = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_images, batches)
        for result in results:
            ids.extend(result[0])
            faces.extend(result[1])
            emotion_labels.extend(result[2])

    return ids, faces, emotion_labels


if __name__ == '__main__':
    multiprocessing.freeze_support()

    path = "data"
    IDs, facedata, emotion_labels = getImageID(path)

    # Save the trained model
    np.savez("Trainer.npz", facedata=facedata, IDs=IDs, emotion_labels=emotion_labels)
    plt.close('all')
    print("Successfully processed the data!")