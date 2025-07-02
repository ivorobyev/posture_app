import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import mediapipe as mp

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
    """Анализирует осанку и возвращает текстовый отчет"""
    h, w, _ = image_shape
    
    # Получаем ключевые точки
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    left_ear = landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
    right_ear = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
    nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    
    # Определяем позу (сидя/стоя)
    sitting = left_hip.y < left_shoulder.y + 0.1 or right_hip.y < right_shoulder.y + 0.1
    
    messages = []
    
    # Проверка наклона головы вперед
    head_forward = (left_ear.y > left_shoulder.y + 0.1 or right_ear.y > right_shoulder.y + 0.1) and \
                   (nose.y > left_shoulder.y or nose.y > right_shoulder.y)
    if head_forward:
        messages.append("• Голова наклонена вперед (текстовая шея)")
    
    # Проверка сутулости
    shoulders_rounded = left_shoulder.x > left_hip.x + 0.05 or right_shoulder.x < right_hip.x - 0.05
    if shoulders_rounded:
        messages.append("• Плечи ссутулены (округлены вперед)")
    
    # Проверка наклона в сторону
    shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
    hip_diff = abs(left_hip.y - right_hip.y)
    if shoulder_diff > 0.05 or hip_diff > 0.05:
        messages.append("• Наклон в сторону (несимметричная осанка)")
    
    # Проверка положения таза
    if sitting and (left_hip.y < left_shoulder.y + 0.15 or right_hip.y < right_shoulder.y + 0.15):
        messages.append("• Таз наклонен вперед (сидя)")
    
    # Формируем итоговый отчет
    if messages:
        report = [
            f"**{'Сидя' if sitting else 'Стоя'} - обнаружены проблемы:**",
            *messages,
            "\n**Рекомендации:**",
            "• Держите голову прямо, уши должны быть над плечами",
            "• Отведите плечи назад и вниз",
            "• Держите спину прямой, избегайте наклонов в стороны",
            "• При сидении опирайтесь на седалищные бугры"
        ]
    else:
        report = [
            f"**Отличная осанка ({'сидя' if sitting else 'стоя'})!**",
            "Все ключевые точки находятся в правильном положении.",
            "\n**Совет:**",
            "• Продолжайте следить за осанкой в течение дня"
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