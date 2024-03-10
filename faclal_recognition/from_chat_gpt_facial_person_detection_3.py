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
    
    label_to_name = {}  # Dicionário para mapear ID da etiqueta para nome
    
    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        name = os.path.splitext(os.path.basename(image_path))[0]  # Extrair nome do arquivo
        label_to_name[idx] = name
        print(name)
        print(label_to_name)
        
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            images.append(image[y:y+h, x:x+w])
            labels.append(idx)
    
    return images, labels, label_to_name
# Carregar imagens, etiquetas e mapeamento de ID para nome
images, labels, label_to_name = load_images_and_labels(dataset_path)

# Treinar o reconhecedor de rostos
face_recognizer.train(images, np.array(label_to_name))

# Salvar o modelo treinado
face_recognizer.save("modelo_reconhecimento_face.xml")

print("Modelo treinado e salvo com sucesso!")
