# python10
# pip install opencv-python mediapipe numpy
import cv2
import mediapipe as mp
import numpy as np

# Cargar la imagen del filtro (máscara completa)
filter_image = cv2.imread('./spiderman.png', cv2.IMREAD_UNCHANGED)

# Inicializar MediaPipe para detección facial y de manos
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Captura de video
cap = cv2.VideoCapture(0)

def is_love_sign_vertical(landmarks):
    """
    Detecta si la mano hace el símbolo de 'amor' en lenguaje de señas en posición vertical.
    """
    thumb_tip = landmarks[4]       # Punta del pulgar
    index_tip = landmarks[8]       # Punta del índice
    middle_tip = landmarks[12]     # Punta del dedo medio
    ring_tip = landmarks[16]       # Punta del anular
    pinky_tip = landmarks[20]      # Punta del meñique

    # Comprobar que el índice y el meñique están levantados verticalmente, con el pulgar extendido lateralmente
    if (index_tip.x < thumb_tip.x and pinky_tip.x > thumb_tip.x) and \
       (index_tip.y < middle_tip.y and pinky_tip.y < ring_tip.y):
        return True
    return False

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
     mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=1) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir la imagen a RGB (MediaPipe trabaja en RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar manos
        hand_results = hands.process(frame_rgb)
        love_sign_detected = False
        
        # Verificar si el gesto de 'amor' fue detectado en posición vertical
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                if is_love_sign_vertical(hand_landmarks.landmark):
                    love_sign_detected = True
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    break  # Solo necesitamos una detección
        
        # Si el gesto fue detectado, aplicar filtro de máscara completa en la cara
        if love_sign_detected:
            # Detección facial
            face_results = face_detection.process(frame_rgb)
            
            if face_results.detections:
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    
                    # Aumentar el tamaño de la máscara un 30% más grande que la cara
                    scale_factor = 2
                    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                    resized_mask = cv2.resize(filter_image, (new_w, new_h))
                    mask_height, mask_width = resized_mask.shape[:2]
                    
                    # Ajustar la posición para centrar la máscara más grande en la cara
                    mask_x = x - (new_w - w) // 2
                    # Ajustar la posición para centrar la máscara más grande en la cara y subirla un poco
                    offset_y = 40  # Cambia este valor para ajustar cuánto quieres subir la máscara
                    mask_y = y - (new_h - h) // 2 - offset_y

                    
                    # Separar canales RGB y alfa de la máscara redimensionada
                    if resized_mask.shape[2] == 4:
                        mask_rgb = resized_mask[:, :, :3]
                        mask_alpha = resized_mask[:, :, 3] / 255.0  # Normalizar el canal alfa
                    else:
                        mask_rgb = resized_mask
                        mask_alpha = np.ones((mask_height, mask_width), dtype=np.float32)  # Alfa completo
                    
                    # Asegurarse de que la máscara se aplique correctamente
                    for i in range(mask_width):
                        for j in range(mask_height):
                            if (0 <= mask_x + i < iw) and (0 <= mask_y + j < ih):
                                alpha = mask_alpha[j, i]
                                frame[mask_y + j, mask_x + i] = (
                                    alpha * mask_rgb[j, i] + (1 - alpha) * frame[mask_y + j, mask_x + i]
                                )
        
        # Mostrar el resultado
        cv2.imshow('Realidad Aumentada', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
