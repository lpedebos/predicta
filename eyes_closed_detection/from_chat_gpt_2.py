import cv2
import dlib
from scipy.spatial import distance as dist
import csv
import time

# Inicializar o detector de rostos e o detector de marcos faciais
detector_faces = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Definir constantes para os índices dos marcos faciais que correspondem aos olhos
EYE_LEFT_START, EYE_LEFT_END = 42, 48
EYE_RIGHT_START, EYE_RIGHT_END = 36, 42

# Função para calcular a razão de aspecto dos olhos
def eye_aspect_ratio(eye):
    # Calcular as distâncias euclidianas entre os dois pares de pontos verticais e horizontais do olho
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Calcular a distância euclidiana entre o par de pontos horizontal do olho
    C = dist.euclidean(eye[0], eye[3])
    # Calcular a razão de aspecto dos olhos
    ear = (A + B) / (2.0 * C)
    return ear

# Iniciar a captura de vídeo
cap = cv2.VideoCapture(0)

# Abrir arquivo CSV para escrita
csv_file = open('olhos_fechados.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Tempo Inicial', 'Tempo Final'])

# Variáveis para acompanhar o tempo inicial e final do fechamento dos olhos
start_time = None
end_time = None
eyes_closed = False

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
    
    # Para cada rosto detectado, detectar olhos
    for face in faces:
        shape = predictor(gray, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        
        # Extrair os pontos dos olhos
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
                    # Escrever o tempo inicial e o tempo final do fechamento dos olhos no arquivo CSV
                    csv_writer.writerow([time.strftime('%H:%M:%S', time.localtime(start_time)),
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
