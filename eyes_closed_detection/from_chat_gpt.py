import cv2
import time
import csv

# Carregar o classificador pré-treinado para detecção de rostos e olhos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Iniciar a captura de vídeo
cap = cv2.VideoCapture(0)

# Variáveis para acompanhar o tempo em que os olhos estão fechados
start_time = None
eyes_closed = False

# Abrir arquivo CSV para escrita
csv_file = open('olhos_fechados.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Tempo de Olhos Fechados'])

while True:
    # Capturar frame por frame
    ret, frame = cap.read()
    
    # Converter o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostos no frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Para cada rosto detectado, detectar olhos
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # Verificar se os olhos estão fechados
        if len(eyes) == 0:
            if start_time is None:
                start_time = time.time()
                cv2.putText(frame, "Olhos Fechados", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif time.time() - start_time > 2:
                eyes_closed = True
                # Escrever o tempo em que os olhos estão fechados no arquivo CSV
                csv_writer.writerow([time.strftime('%H:%M:%S', time.localtime(start_time))])
        else:
            start_time = None
            eyes_closed = False
    
    # Mostrar o frame
    cv2.imshow('frame',frame)
    
    # Sair do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fechar o arquivo CSV e liberar a captura
csv_file.close()
cap.release()
cv2.destroyAllWindows()
