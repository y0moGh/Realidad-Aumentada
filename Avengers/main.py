import cv2
import mediapipe as mp
import numpy as np

# Cargar imágenes de filtros
filter_spiderman = cv2.imread('./spiderman.png', cv2.IMREAD_UNCHANGED)
filter_thanos = cv2.imread('./thanos_2.png', cv2.IMREAD_UNCHANGED)
filter_thanos_2 = cv2.imread('./thanos.png', cv2.IMREAD_UNCHANGED)  # Nuevo guante para la mano derecha
filter_ironman = cv2.imread('./ironman_mask.png', cv2.IMREAD_UNCHANGED)
filter_ironman_glove = cv2.imread('./ironman_hand.png', cv2.IMREAD_UNCHANGED)
filter_thanos_face = cv2.imread('./thanos_face.png', cv2.IMREAD_UNCHANGED)  # Nuevo filtro para la cara de Thanos

# Verificar si las imágenes fueron cargadas correctamente
if filter_spiderman is None:
    print("Error: No se pudo cargar la imagen 'spiderman.png'. Verifica la ruta.")
    raise FileNotFoundError("Falta el filtro Spiderman.")
if filter_thanos is None:
    print("Error: No se pudo cargar la imagen 'thanos.png'. Verifica la ruta.")
    raise FileNotFoundError("Falta el filtro Thanos.")
if filter_thanos_2 is None:
    print("Error: No se pudo cargar la imagen 'thanos_2.png'. Verifica la ruta.")
    raise FileNotFoundError("Falta el filtro Thanos para la mano derecha.")
if filter_ironman is None:
    print("Error: No se pudo cargar la imagen 'ironman_mask.png'. Verifica la ruta.")
    raise FileNotFoundError("Falta la máscara de Ironman.")
if filter_ironman_glove is None:
    print("Error: No se pudo cargar la imagen 'ironman_hand.png'. Verifica la ruta.")
    raise FileNotFoundError("Falta el guante de Ironman.")
if filter_thanos_face is None:
    print("Error: No se pudo cargar la imagen 'thanos_face.png'. Verifica la ruta.")
    raise FileNotFoundError("Falta la cara de Thanos.")

# Inicializar MediaPipe para detección facial y de manos
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands

# Captura de video
cap = cv2.VideoCapture(0)

# Configurar ventana en pantalla completa
cv2.namedWindow('Realidad Aumentada', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Realidad Aumentada', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Función para verificar el gesto de Spiderman (índice y meñique extendidos)
def is_spiderman_gesture(landmarks):
    index_tip = landmarks[8]
    pinky_tip = landmarks[20]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    return (index_tip.y < middle_tip.y and pinky_tip.y < ring_tip.y and 
            middle_tip.y > index_tip.y and ring_tip.y > pinky_tip.y)

# Función para verificar el gesto del chasquido (pulgar y medio juntos) - Gesto de Thanos
def is_snap_gesture(landmarks):
    thumb_tip = landmarks[4]
    middle_tip = landmarks[12]
    return abs(thumb_tip.x - middle_tip.x) < 0.05 and abs(thumb_tip.y - middle_tip.y) < 0.05

# Función para verificar si la mano está abierta (todos los dedos extendidos) - Gesto de Iron Man
def is_open_hand_gesture(landmarks):
    fingers = [8, 12, 16, 20]  # puntas de los dedos índice, medio, anular, meñique
    for finger_tip in fingers:
        if landmarks[finger_tip].y > landmarks[finger_tip - 2].y:
            return False
    return True

# Función para rotar una imagen
def rotate_image(image, angle): 
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    return rotated

def overlay_filter(frame, filter_img, x, y, scale_factor=1.0):
    filter_h, filter_w = filter_img.shape[:2]
    new_w = int(filter_w * scale_factor)
    new_h = int(filter_h * scale_factor)
    resized_filter = cv2.resize(filter_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    y1, y2 = max(0, y), min(frame.shape[0], y + new_h)
    x1, x2 = max(0, x), min(frame.shape[1], x + new_w)

    filter_y1, filter_y2 = max(0, -y), min(new_h, frame.shape[0] - y)
    filter_x1, filter_x2 = max(0, -x), min(new_w, frame.shape[1] - x)

    if filter_y1 < filter_y2 and filter_x1 < filter_x2:
        filter_region = resized_filter[filter_y1:filter_y2, filter_x1:filter_x2]
        alpha = filter_region[:, :, 3] / 255.0
        for c in range(3):  # BGR channels
            frame[y1:y2, x1:x2, c] = (
                alpha * filter_region[:, :, c] +
                (1 - alpha) * frame[y1:y2, x1:x2, c]
            ).astype(np.uint8)

# Iniciar detección de rostros y manos
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
     mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=2, model_complexity=1) as hands:  # max_num_hands=2 para detectar ambas manos

    print("Iniciando detección. Presiona 'q' para salir.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectar manos
        hand_results = hands.process(frame_rgb)
        active_filter = None
        filter_position = None
        apply_to_face = False
        ironman_mode = False
        thanos_face_mode = False  # Para aplicar la cara de Thanos

        # Procesar gestos de las manos
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                hand_label = hand_results.multi_handedness[idx].classification[0].label
                # Detectar gesto de Spiderman
                if is_spiderman_gesture(hand_landmarks.landmark):
                    active_filter = filter_spiderman
                    apply_to_face = True  # Se aplicará el filtro al rostro
                    break
                # Detectar gesto de Thanos (chasquido)
                elif is_snap_gesture(hand_landmarks.landmark):
                    if hand_label == 'Left':
                        active_filter = filter_thanos  # Guante de Thanos para la mano izquierda
                    else:
                        active_filter = filter_thanos_2  # Guante alternativo para la mano derecha
                    filter_position = hand_landmarks.landmark[9]  # Centro aproximado de la palma de la mano
                    thanos_face_mode = True  # Aplicar cara de Thanos
                    break

                # Detectar gesto de Iron Man (mano abierta)
                elif is_open_hand_gesture(hand_landmarks.landmark):
                    ironman_mode = True
                    filter_position = hand_landmarks.landmark[9]
                    break

        # Aplicar filtros de Iron Man (máscara en la cara y guante en la mano)
        if ironman_mode:
            # Aplicar máscara de Iron Man al rostro
            face_results = face_detection.process(frame_rgb)
            if face_results.detections:
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    scale_factor = 0.6
                    overlay_filter(frame, filter_ironman, x - 57, y - 100, scale_factor)

            # Aplicar guante de Iron Man en la mano
            ih, iw, _ = frame.shape
            x = int(filter_position.x * iw)
            y = int(filter_position.y * ih)
            overlay_filter(frame, filter_ironman_glove, x - filter_ironman_glove.shape[1] // 4, y - filter_ironman_glove.shape[0] // 4, scale_factor=0.5)
        
        
        # Aplicar filtro de Spiderman o Thanos si se detecta el gesto correspondiente
        elif active_filter is not None:
            # Si es el filtro de Spiderman, aplicarlo en el rostro
            if apply_to_face:
                face_results = face_detection.process(frame_rgb)
                if face_results.detections:
                    for detection in face_results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        x = int(bboxC.xmin * iw)
                        y = int(bboxC.ymin * ih)
                        scale_factor = 0.37
                        overlay_filter(frame, active_filter, x - 55, y - 90, scale_factor)
            # Aplicar filtro de Thanos en la mano
            else:
                ih, iw, _ = frame.shape
                x = int(filter_position.x * iw)
                y = int(filter_position.y * ih)
                if active_filter is filter_thanos_2:
                    x -= 80  # Ajusta el valor (en píxeles) para mover más o menos hacia la izquierda
                    y -= 30
                else:
                    x -= 10
                    y -= 20

                overlay_filter(frame, active_filter, x - active_filter.shape[1] // 4, y - active_filter.shape[0] // 4, scale_factor=0.7)

        # Aplicar la cara de Thanos si está activada
        if thanos_face_mode:
            face_results = face_detection.process(frame_rgb)
            if face_results.detections:
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    scale_factor = 0.35
                    overlay_filter(frame, filter_thanos_face, x + 10, y - 100, scale_factor)

        # Mostrar el resultado
        cv2.imshow('Realidad Aumentada', frame)

        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
