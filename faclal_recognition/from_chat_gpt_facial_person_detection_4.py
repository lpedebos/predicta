import cv2
import os
import numpy as np
import csv

# Diretório contendo as imagens das pessoas
dataset_path = 'dataset'

# Caminho para o arquivo CSV contendo o mapeamento de nome de pessoa para imagem
csv_file = 'dataset/person_image_dict.csv'

# Inicializar o detector de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar o reconhecedor de rostos LBPH
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Função para carregar imagens e etiquetas do dataset
def load_images_and_labels(dataset_path, csv_file):
    images = []
    labels = []
    label_to_name = {}  # Dicionário para mapear ID da etiqueta para nome
    
    # Carregar mapeamento de nome de pessoa para imagem do CSV
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            name = row['nome']
            image_name = row['imagem']
            image_path = os.path.join(dataset_path, image_name)
            
            if os.path.isfile(image_path):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
                    for (x, y, w, h) in faces:
                        images.append(image[y:y+h, x:x+w])
                        labels.append(len(label_to_name))  # Usar o ID incremental
                    label_to_name[len(label_to_name)] = name

            else:
                print(f"Arquivo de imagem '{image_name}' não encontrado.")
    
    return images, labels, label_to_name

# Carregar imagens, etiquetas e mapeamento de ID para nome
images, labels, label_to_name = load_images_and_labels(dataset_path, csv_file)

# Treinar o reconhecedor de rostos
face_recognizer.train(images, np.array(labels))

# Salvar o modelo treinado
face_recognizer.save("modelo_reconhecimento_face.xml")

print("Modelo treinado e salvo com sucesso!")
