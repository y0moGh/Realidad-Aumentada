# python10
# pip install opencv-python mediapipe numpy
import cv2
import mediapipe as mp
import numpy as np

# Cargar imágenes de filtros
filter_spiderman = cv2.imread('./spiderman.png', cv2.IMREAD_UNCHANGED)
filter_thanos = cv2.imread('./thanos.png', cv2.IMREAD_UNCHANGED)

# Inicializar MediaPipe para detección facial y de manos
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Captura de video
cap = cv2.VideoCapture(0)

def is_love_sign_vertical(landmarks):
    # Detectar el símbolo de amor en posición vertical
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    return (index_tip.x < thumb_tip.x and pinky_tip.x > thumb_tip.x) and \
           (index_tip.y < middle_tip.y and pinky_tip.y < ring_tip.y)

def is_snap_gesture(landmarks):
    # Detectar el gesto de chasquido
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    return (abs(thumb_tip.x - middle_tip.x) < 0.02 and
            abs(thumb_tip.y - middle_tip.y) < 0.02 and
            index_tip.y < middle_tip.y)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
     mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=1) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir la imagen a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar manos
        hand_results = hands.process(frame_rgb)
        active_filter = None
        filter_position = None
        apply_to_face = False

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                if is_love_sign_vertical(hand_landmarks.landmark):
                    active_filter = filter_spiderman
                    apply_to_face = True
                    break  # No verificar más si encontramos el filtro
                elif is_snap_gesture(hand_landmarks.landmark):
                    active_filter = filter_thanos
                    filter_position = hand_landmarks.landmark[0]
                    break

        if active_filter is not None:
            if apply_to_face:
                # Detección facial para aplicar el filtro de Spiderman
                face_results = face_detection.process(frame_rgb)
                
                if face_results.detections:
                    for detection in face_results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                        
                        scale_factor = 2
                        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                        resized_filter = cv2.resize(filter_spiderman, (new_w, new_h))
                        filter_height, filter_width = resized_filter.shape[:2]
                        
                        filter_x = x - (new_w - w) // 2
                        filter_y = y - (new_h - h) // 2 - 40

                        if resized_filter.shape[2] == 4:
                            filter_rgb = resized_filter[:, :, :3]
                            filter_alpha = resized_filter[:, :, 3] / 255.0
                        else:
                            filter_rgb = resized_filter
                            filter_alpha = np.ones((filter_height, filter_width), dtype=np.float32)

                        for i in range(filter_width):
                            for j in range(filter_height):
                                if (0 <= filter_x + i < iw) and (0 <= filter_y + j < ih):
                                    alpha = filter_alpha[j, i]
                                    frame[filter_y + j, filter_x + i] = (
                                        alpha * filter_rgb[j, i] + (1 - alpha) * frame[filter_y + j, filter_x + i]
                                    )
            else:
                # Aplicar el filtro de Thanos en la mano
                ih, iw, _ = frame.shape
                x, y = int(filter_position.x * iw), int(filter_position.y * ih)
                
                scale_factor = 1.7
                new_w, new_h = int(100 * scale_factor), int(100 * scale_factor)
                resized_filter = cv2.resize(filter_thanos, (new_w, new_h))
                filter_height, filter_width = resized_filter.shape[:2]
                
                filter_x = x - filter_width // 2
                filter_y = y - filter_height // 2 - 50

                if resized_filter.shape[2] == 4:
                    filter_rgb = resized_filter[:, :, :3]
                    filter_alpha = resized_filter[:, :, 3] / 255.0
                else:
                    filter_rgb = resized_filter
                    filter_alpha = np.ones((filter_height, filter_width), dtype=np.float32)

                for i in range(filter_width):
                    for j in range(filter_height):
                        if (0 <= filter_x + i < iw) and (0 <= filter_y + j < ih):
                            alpha = filter_alpha[j, i]
                            frame[filter_y + j, filter_x + i] = (
                                alpha * filter_rgb[j, i] + (1 - alpha) * frame[filter_y + j, filter_x + i]
                            )
        
        # Mostrar el resultado
        cv2.imshow('Realidad Aumentada', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
