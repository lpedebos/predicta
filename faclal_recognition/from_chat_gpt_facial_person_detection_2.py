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
    
    label_id = 0  # Inicializar ID da etiqueta
    label_to_id = {}  # Dicionário para mapear nome da pessoa para ID da etiqueta
    
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = os.path.splitext(os.path.basename(image_path))[0].split('_')[0]  # Extrair apenas o nome base
        
        # Verificar se o nome já foi mapeado para um ID
        if label not in label_to_id:
            label_to_id[label] = label_id
            label_id += 1
        
        # Adicionar a imagem e a etiqueta (ID) correspondente
        images.append(image)
        labels.append(label_to_id[label])
    
    return images, labels

# Carregar imagens e etiquetas
images, labels = load_images_and_labels(dataset_path)

# Treinar o reconhecedor de rostos
face_recognizer.train(images, np.array(labels))

# Salvar o modelo treinado
face_recognizer.save("modelo_reconhecimento_face.xml")

print("Modelo treinado e salvo com sucesso!")
