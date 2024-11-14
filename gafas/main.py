import cv2  # pip install opencv-python
import mediapipe as mp
import numpy as np

# Ruta de la imagen del filtro
filter_path = 'D:\JuaniSchoolWork\Realidad-Aumentada\gafas\gafas.png'

# Cargar la imagen del filtro (gafas)
filter_image = cv2.imread(filter_path, cv2.IMREAD_UNCHANGED)

# Verificar si la imagen fue cargada correctamente
if filter_image is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen desde la ruta: {filter_path}")

# Redimensionar el filtro según sea necesario
filter_image = cv2.resize(filter_image, (150, 50))  # Ajustar tamaño según sea necesario
filter_height, filter_width = filter_image.shape[:2]

# Separar los canales de color y el canal alfa (transparencia)
if filter_image.shape[2] == 4:  # Imagen con canal alfa
    filter_rgb = filter_image[:, :, :3]  # Canales RGB
    filter_alpha = filter_image[:, :, 3]  # Canal Alfa
else:
    filter_rgb = filter_image  # Si no tiene canal alfa, usar la imagen tal cual
    filter_alpha = np.ones((filter_height, filter_width), dtype=np.uint8) * 255  # Canal alfa completo (opaco)

# Inicializar MediaPipe para detección facial
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertir la imagen a RGB (MediaPipe trabaja en RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar la imagen con MediaPipe para detección de rostros
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        results = face_detection.process(frame_rgb)
        
        # Dibujar los resultados
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # Calcular la posición del filtro (gafas) sobre los ojos
                filter_x = x + w // 2 - filter_width // 2  # Centrar horizontalmente
                filter_y = y + int(h * 0.14)  # Aproximar altura de los ojos (25% de la altura de la cara)

                # Asegurarse de que el filtro se aplique correctamente
                for i in range(filter_width):
                    for j in range(filter_height):
                        if (0 <= filter_x + i < iw) and (0 <= filter_y + j < ih):
                            alpha = filter_alpha[j, i] / 255.0  # Normalizar canal alfa
                            # Mezclar píxeles del filtro y del video
                            frame[filter_y + j, filter_x + i] = (
                                alpha * filter_rgb[j, i] + (1 - alpha) * frame[filter_y + j, filter_x + i]
                            )

    # Mostrar el resultado
    cv2.imshow('Realidad Aumentada', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
