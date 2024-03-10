import cv2
import os
import numpy as np
import csv

# Diretório contendo as imagens das pessoas
dataset_path = 'dataset'

# Caminho para o arquivo CSV contendo o mapeamento de nome de pessoa para imagem
csv_file = dataset_path + '/person_image_dict.csv'

# Inicializar o detector de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar o reconhecedor de rostos LBPH
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Função para carregar imagens e etiquetas do dataset
def load_images_and_labels(dataset_path, csv_file):
    images = []
    labels = []
    label_to_name = {}  # Dicionário para mapear ID da etiqueta para nome
    
    # Ler o arquivo CSV e criar um dicionário para mapear o nome do arquivo para o nome da pessoa
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            name = row['nome']
            image_prefix = row['imagem']
            for filename in os.listdir(dataset_path):
                if filename.startswith(image_prefix) and filename.lower().endswith('.jpg'):
                    image_path = os.path.join(dataset_path, filename)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        images.append(image)
                        if image_prefix not in label_to_name:
                            label_to_name[image_prefix] = len(label_to_name)  # Atribuímos um ID único para cada prefixo
                        labels.append(label_to_name[image_prefix])  # Usamos o ID da etiqueta
    labels = np.array(labels)  # Converter a lista de etiquetas para um array numpy
    return images, labels, label_to_name



# Carregar imagens, etiquetas e mapeamento de ID para nome
images, labels, id_to_name = load_images_and_labels(dataset_path, csv_file)

# Treinar o reconhecedor de rostos
face_recognizer.train(images, np.array(labels))

# Salvar o modelo treinado
face_recognizer.save("modelo_reconhecimento_face.xml")

print("Modelo treinado e salvo com sucesso!")
