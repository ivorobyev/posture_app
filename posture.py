import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import mediapipe as mp

# Инициализация MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)
mp_drawing = mp.solutions.drawing_utils

def analyze_posture(image):
    """Анализирует осанку на изображении и возвращает изображение с ключевыми точками и текст с анализом"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    annotated_image = image.copy()
    if results.pose_landmarks:
        # Рисуем ключевые точки (без текста)
        mp_drawing.draw_landmarks(
            annotated_image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
        posture_status = check_posture(results.pose_landmarks, image.shape)
    else:
        posture_status = "Ключевые точки не обнаружены"
    
    return annotated_image, posture_status

def check_posture(landmarks, image_shape):
    h, w, _ = image_shape

    # Ключевые точки для анализа
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    left_ear = landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
    right_ear = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
    nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

    messages = []

    # Определение положения тела: сидит или стоит
    # Предположим, что если таз значительно ниже плеч — сидит
    avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    avg_hip_y = (left_hip.y + right_hip.y) / 2
    sitting = avg_hip_y - avg_shoulder_y > 0.15

    # Проверка наклона головы вперед (текстовая шея)
    head_forward = False
    if (left_ear.y > avg_shoulder_y + 0.07 or right_ear.y > avg_shoulder_y + 0.07) and \
       (nose.y > avg_shoulder_y):
        head_forward = True
        messages.append("• Голова наклонена вперед (текстовая шея)")

    # Проверка сутулости плеч
    # Рассчитаем угол между плечами и тазом
    shoulder_dx = right_shoulder.x - left_shoulder.x
    shoulder_dy = right_shoulder.y - left_shoulder.y
    hip_dx = right_hip.x - left_hip.x
    hip_dy = right_hip.y - left_hip.y

    shoulder_slope = abs(shoulder_dy)
    hip_slope = abs(hip_dy)

    # Сутулость — когда плечи сдвинуты вперед относительно таза по X (по горизонтали)
    shoulders_forward = (left_shoulder.x > left_hip.x + 0.05) or (right_shoulder.x < right_hip.x - 0.05)
    if shoulders_forward:
        messages.append("• Плечи ссутулены (округлены вперед)")

    # Проверка наклона тела в сторону (асимметрия по вертикали)
    if shoulder_slope > 0.05:
        messages.append("• Плечи не на одном уровне (наклон в сторону)")
    if hip_slope > 0.05:
        messages.append("• Таз не на одном уровне (наклон в сторону)")

    # Проверка наклона головы вбок (неравенство ушей по вертикали)
    ear_diff = abs(left_ear.y - right_ear.y)
    if ear_diff > 0.04:
        messages.append("• Голова наклонена в сторону")

    # Проверка сгиба локтей — важна для оценки положения рук
    def angle_between_points(a, b, c):
        import math
        # угол в градусах между точками a-b-c в плоскости XY
        ba = (a.x - b.x, a.y - b.y)
        bc = (c.x - b.x, c.y - b.y)
        dot_product = ba[0]*bc[0] + ba[1]*bc[1]
        mag_ba = (ba[0]**2 + ba[1]**2)**0.5
        mag_bc = (bc[0]**2 + bc[1]**2)**0.5
        if mag_ba * mag_bc == 0:
            return 0
        cos_angle = dot_product / (mag_ba * mag_bc)
        cos_angle = max(min(cos_angle,1),-1)  # ограничить диапазон
        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)

    left_elbow_angle = angle_between_points(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = angle_between_points(right_shoulder, right_elbow, right_wrist)

    # Если локти сильно согнуты — возможно неправильная поза за рабочим столом
    if left_elbow_angle < 60 or right_elbow_angle < 60:
        messages.append("• Локти сильно согнуты (возможна напряженность рук)")

    # Проверка коленей — согнуты или выпрямлены
    # Аналогично локтям — для простоты, можно добавить, если надо

    # Итоговые рекомендации
    if not messages:
        report = [
            f"**Отличная осанка ({'сидя' if sitting else 'стоя'})!**",
            "Все ключевые точки находятся в правильном положении.",
            "\n**Совет:**",
            "• Продолжайте следить за осанкой в течение дня"
        ]
    else:
        report = [
            f"**{'Сидя' if sitting else 'Стоя'} - обнаружены проблемы:**",
            *messages,
            "\n**Рекомендации:**",
            "• Держите голову прямо, уши должны быть над плечами",
            "• Отведите плечи назад и вниз",
            "• Держите спину прямой, избегайте наклонов в стороны",
            "• При сидении опирайтесь на седалищные бугры"
        ]

    return "\n\n".join(report)

def main():
    st.set_page_config(layout="wide")
    st.title("📷 Анализатор осанки с веб-камеры")
    st.write("Это приложение анализирует вашу осанку в реальном времени")
    
    # Создаем две колонки
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Вид с камеры")
        run = st.checkbox("Включить веб-камеру", value=True)
        FRAME_WINDOW = st.image([])
    
    with col2:
        st.header("Анализ осанки")
        status_placeholder = st.empty()
        if not run:
            status_placeholder.markdown("""
                **Ожидание данных с камеры...**
                
                Включите веб-камеру для анализа осанки.
            """)
    
    camera = cv2.VideoCapture(0)
    
    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Анализ осанки
        analyzed_frame, posture_status = analyze_posture(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        analyzed_frame = cv2.cvtColor(analyzed_frame, cv2.COLOR_BGR2RGB)
        
        # Отображение в колонках
        with col1:
            FRAME_WINDOW.image(analyzed_frame)
        
        with col2:
            status_placeholder.markdown(posture_status)
        
        time.sleep(0.1)
    
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()