import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Кэшируем загрузку модели
@st.cache_resource
def load_model():
    return YOLO('yolov8n-pose.pt')

model = load_model()

def get_keypoint_safe(keypoints, idx, fallback=None):
    """Безопасное получение точки с проверкой confidence"""
    if keypoints is None or idx >= len(keypoints):
        return fallback if fallback is not None else np.array([0, 0, 0])
    point = keypoints[idx]
    if len(point) >= 3 and point[2] > 0.25:  # confidence > 0.25
        return point[:3]
    return fallback if fallback is not None else np.array([0, 0, 0])

def analyze_posture(image):
    """Анализ осанки с защитой от ошибок"""
    try:
        results = model(image, verbose=False)
        
        if not results or len(results) == 0:
            return image, "Встаньте перед камерой"
        
        annotated_image = results[0].plot()
        keypoints = results[0].keypoints.xy[0].cpu().numpy() if results[0].keypoints else None
        
        if keypoints is None or len(keypoints) < 6:  # Минимум: голова и плечи
            return annotated_image, "Повернитесь к камере лицом"
        
        # COCO-формат ключевых точек
        NOSE = 0
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        LEFT_EAR = 3
        RIGHT_EAR = 4
        
        # Получаем точки
        ls = get_keypoint_safe(keypoints, LEFT_SHOULDER)
        rs = get_keypoint_safe(keypoints, RIGHT_SHOULDER)
        le = get_keypoint_safe(keypoints, LEFT_EAR, ls)
        re = get_keypoint_safe(keypoints, RIGHT_EAR, rs)
        nose = get_keypoint_safe(keypoints, NOSE, (ls + rs) / 2)
        
        # Проверяем минимальный набор точек
        if ls[2] < 0.2 or rs[2] < 0.2:
            return annotated_image, "Встаньте прямо (плечи не видны)"
        
        # Анализ осанки
        messages = []
        
        # 1. Наклон головы вперед
        head_forward = (nose[1] > (ls[1] + rs[1])/2 + 0.1*image.shape[0])
        if head_forward:
            messages.append("• Голова наклонена вперед")
        
        # 2. Сутулость плеч
        shoulders_rounded = (ls[0] > rs[0] + 0.1*image.shape[1]) or \
                          (rs[0] < ls[0] - 0.1*image.shape[1])
        if shoulders_rounded:
            messages.append("• Плечи ссутулены")
        
        # Формируем отчет
        if messages:
            report = [
                "**Обнаружены проблемы с осанкой:**",
                *messages,
                "\n**Рекомендации:**",
                "• Держите голову прямо",
                "• Расправьте плечи",
                "• Следите за положением спины"
            ]
        else:
            report = ["**Отличная осанка!**"]
        
        return annotated_image, "\n".join(report)
        
    except Exception as e:
        return image, f"Ошибка анализа: {str(e)}"

def main():
    st.set_page_config(layout="wide")
    st.title("📷 Анализатор осанки")
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
    
    if run:
        camera = cv2.VideoCapture(0)
        try:
            while run:
                ret, frame = camera.read()
                if not ret:
                    st.error("Ошибка чтения камеры")
                    break
                
                # Анализ осанки
                analyzed_frame, posture_status = analyze_posture(frame)
                
                # Отображение в колонках
                with col1:
                    FRAME_WINDOW.image(analyzed_frame, channels="BGR")
                
                with col2:
                    status_placeholder.markdown(posture_status)
                
                time.sleep(0.1)
        finally:
            camera.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()