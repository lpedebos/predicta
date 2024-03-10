import cv2
import os
import numpy as np

# Diretório contendo as imagens das pessoas
dataset_path = 'dataset'

# Inicializar o detector de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar o reconhecedor de rostos LBPH
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Função para carregar imagens e etiquetas do dataset
def load_images_and_labels(dataset_path):
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
    images = []
    labels = []
    
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = int(os.path.split(image_path)[1].split('.')[0].replace("person", ""))
        
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            images.append(image[y:y+h, x:x+w])
            labels.append(label)
    
    return images, labels

# Carregar imagens e etiquetas
images, labels = load_images_and_labels(dataset_path)

# Treinar o reconhecedor de rostos
face_recognizer.train(images, np.array(labels))

# Salvar o modelo treinado
face_recognizer.save("modelo_reconhecimento_face.xml")

print("Modelo treinado e salvo com sucesso!")
