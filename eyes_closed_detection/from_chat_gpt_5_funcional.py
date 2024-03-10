import cv2
import dlib
import numpy as np
import csv
import time
from scipy.spatial import distance as dist

# Inicializar o detector de rostos e o detector de marcos faciais
detector_faces = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()  # Usar o reconhecedor LBPH

# Carregar o modelo treinado para reconhecimento facial
face_recognizer.read("modelo_reconhecimento_face.xml")

# Dicionário para mapear IDs de rostos reconhecidos para nomes
id_to_name = {}  # Inicialmente vazio, será atualizado dinamicamente

# Definir constantes para os índices dos marcos faciais que correspondem aos olhos
EYE_LEFT_START, EYE_LEFT_END = 42, 48
EYE_RIGHT_START, EYE_RIGHT_END = 36, 42

# Função para calcular a razão de aspecto dos olhos
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Iniciar a captura de vídeo
cap = cv2.VideoCapture(0)

# Abrir arquivo CSV para escrita
csv_file = open('olhos_fechados.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Nome', 'Tempo Inicial', 'Tempo Final'])

# Variáveis para acompanhar o tempo inicial e final do fechamento dos olhos
start_time = None
end_time = None
eyes_closed = False

# Função para obter nome correspondente ao ID
def get_name_from_id(id_):
    return id_to_name.get(id_, f"Pessoa_{id_}")

while True:
    # Capturar frame por frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Redimensionar o frame para melhorar o desempenho
    frame = cv2.resize(frame, (640, 480))
    
    # Converter o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostos no frame
    faces = detector_faces(gray, 0)
    
    # Limpar dicionário de IDs para nomes antes de preencher novamente
    id_to_name.clear()
    
    # Definir margem de segurança como 10% do tamanho da imagem em cada lado
    margin = int(0.1 * min(frame.shape[0], frame.shape[1]))
    
    # Para cada rosto detectado, detectar olhos
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        # Verificar se o rosto está dentro da margem de segurança
        if x < margin or y < margin or x + w > frame.shape[1] - margin or y + h > frame.shape[0] - margin:
            continue
        
        # Extrair a região de interesse (ROI) do rosto para o reconhecimento facial
        roi_gray = gray[y:y+h, x:x+w]
        id_, conf = face_recognizer.predict(roi_gray)
        
        # Verificar a confiança da previsão
        if conf >= 45 and conf <= 85:
            # Extrair o nome da pessoa correspondente ao ID
            name = get_name_from_id(id_)
            
            # Adicionar o nome à lista de nomes se ainda não estiver lá
            if id_ not in id_to_name:
                id_to_name[id_] = name
            
            # Calcular as coordenadas do retângulo
            rect_x1 = x
            rect_y1 = y
            rect_x2 = x + w
            rect_y2 = y + h
            
            # Desenhar um retângulo ao redor do rosto
            cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 0, 0), 2)
                
            # Calcular as coordenadas do texto
            text_x = x
            text_y = y - 10
            text_x = max(text_x, 0)
            text_y = max(text_y, 0)
                
            # Exibir o nome da pessoa acima do retângulo
            cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
            # Extrair os pontos dos olhos
            shape = predictor(gray, face)
            shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
            left_eye = shape[EYE_LEFT_START:EYE_LEFT_END]
            right_eye = shape[EYE_RIGHT_START:EYE_RIGHT_END]
            
            # Calcular a razão de aspecto dos olhos
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            
            # Calcular a média da razão de aspecto dos olhos
            ear = (left_ear + right_ear) / 2.0
            
            # Verificar se a razão de aspecto indica que os olhos estão fechados
            if ear < 0.25:
                if not eyes_closed:
                    start_time = time.time()
                    eyes_closed = True
            else:
                if eyes_closed:
                    end_time = time.time()
                    if end_time - start_time >= 2:
                        # Escrever o nome, tempo inicial e tempo final do fechamento dos olhos no arquivo CSV
                        csv_writer.writerow([name, time.strftime('%H:%M:%S', time.localtime(start_time)),
                                             time.strftime('%H:%M:%S', time.localtime(end_time))])
                eyes_closed = False
    
    # Mostrar o frame
    cv2.imshow('frame', frame)
    
    # Sair do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fechar o arquivo CSV, liberar a captura e fechar todas as janelas
csv_file.close()
cap.release()
cv2.destroyAllWindows()
