import cv2
import os
import numpy as np
import csv

# Inicializar o detector de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar o reconhecedor LBPH
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Função para carregar imagens e etiquetas do dataset
def load_images_and_labels(dataset_path):
    images = []
    labels = []
    label_to_name = {}
    label = 0
    
    # Iterar sobre as pastas de pessoas no dataset
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue
        
        # Mapear o ID da etiqueta para o nome da pessoa
        label_to_name[label] = person_name
        
        # Iterar sobre as imagens da pessoa
        for file_name in os.listdir(person_path):
            if not file_name.endswith('.jpg'):
                continue
            image_path = os.path.join(person_path, file_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            images.append(image)
            labels.append(label)
        
        label += 1
    
    return images, labels, label_to_name

# Diretório contendo as pastas das pessoas
dataset_path = 'dataset copy'

# Carregar imagens, etiquetas e mapeamento de ID para nome
images, labels, label_to_name = load_images_and_labels(dataset_path)

# Treinar o reconhecedor de rostos
face_recognizer.train(images, np.array(labels))

# Salvar o modelo treinado
face_recognizer.save("modelo_reconhecimento_face.xml")

# Salvar o mapeamento de ID para nome em um arquivo CSV
with open('label_to_name.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for label, name in label_to_name.items():
        writer.writerow([label, name])

print("Modelo treinado e salvo com sucesso!")
