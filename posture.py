import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
from enum import Enum

class PostureType(Enum):
    STANDING = "стоя"
    SITTING = "сидя"
    UNKNOWN = "не определено"

@st.cache_resource
def load_model():
    return YOLO('yolov8n-pose.pt')

model = load_model()

def get_keypoint_safe(keypoints, idx, fallback=None):
    """Безопасное получение точки с проверкой confidence"""
    if keypoints is None or idx >= len(keypoints):
        return fallback if fallback is not None else np.array([0, 0, 0])
    
    point = keypoints[idx]
    if len(point) >= 3 and point[2] > 0.25:  # Более низкий порог confidence
        return point[:3]
    return fallback if fallback is not None else np.array([0, 0, 0])

def calculate_angle(a, b, c):
    """Вычисление угла между тремя точками"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cosine_angle))

def determine_posture_type(keypoints, img_height):
    """Определение типа позы (стоя/сидя)"""
    # COCO-формат ключевых точек
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    ls = get_keypoint_safe(keypoints, LEFT_SHOULDER)
    rs = get_keypoint_safe(keypoints, RIGHT_SHOULDER)
    lh = get_keypoint_safe(keypoints, LEFT_HIP)
    rh = get_keypoint_safe(keypoints, RIGHT_HIP)
    lk = get_keypoint_safe(keypoints, LEFT_KNEE)
    rk = get_keypoint_safe(keypoints, RIGHT_KNEE)
    
    # Если видно колени - анализируем их положение
    if lk[2] > 0.3 or rk[2] > 0.3:
        hip_knee_ratio = ((lh[1] - lk[1]) + (rh[1] - rk[1])) / (2 * img_height)
        if hip_knee_ratio < 0.15:  # Колени близко к бедрам
            return PostureType.SITTING
        return PostureType.STANDING
    
    # Альтернативные критерии, если колени не видны
    shoulder_hip_ratio = ((ls[1] - lh[1]) + (rs[1] - rh[1])) / (2 * img_height)
    if shoulder_hip_ratio < 0.1:  # Плечи близко к бедрам
        return PostureType.SITTING
    
    return PostureType.UNKNOWN

def analyze_posture(image):
    """Комплексный анализ осанки с учетом ракурса"""
    try:
        results = model(image, verbose=False)
        
        if not results or len(results) == 0 or results[0].keypoints is None:
            return image, "Встаньте перед камерой", PostureType.UNKNOWN
        
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        annotated_image = results[0].plot()
        
        if len(keypoints) < 17:  # Проверяем количество точек COCO
            return annotated_image, "Недостаточно точек для анализа", PostureType.UNKNOWN
        
        # Определяем тип позы
        posture_type = determine_posture_type(keypoints, image.shape[0])
        
        # Основные точки для анализа
        NOSE = 0
        LEFT_EAR = 3
        RIGHT_EAR = 4
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        LEFT_HIP = 11
        RIGHT_HIP = 12
        
        # Получаем точки с fallback-значениями
        ls = get_keypoint_safe(keypoints, LEFT_SHOULDER)
        rs = get_keypoint_safe(keypoints, RIGHT_SHOULDER)
        lh = get_keypoint_safe(keypoints, LEFT_HIP)
        rh = get_keypoint_safe(keypoints, RIGHT_HIP)
        le = get_keypoint_safe(keypoints, LEFT_EAR, ls)
        re = get_keypoint_safe(keypoints, RIGHT_EAR, rs)
        nose = get_keypoint_safe(keypoints, NOSE, (ls + rs) / 2)
        
        # Проверяем минимальный набор точек
        if ls[2] < 0.2 or rs[2] < 0.2:
            return annotated_image, "Не вижу плечи - встаньте прямо", posture_type
        
        messages = []
        
        # 1. Анализ головы (для всех ракурсов)
        head_forward = False
        if le[2] > 0.25 and re[2] > 0.25:  # Если видны оба уха
            head_angle = calculate_angle(le[:2], nose[:2], re[:2])
            if head_angle < 150:  # Наклон головы вперед
                head_forward = True
        elif nose[2] > 0.3:
            if nose[1] > (ls[1] + rs[1])/2 + 0.1*image.shape[0]:  # Нос ниже плеч
                head_forward = True
        
        if head_forward:
            messages.append("• Голова наклонена вперед")
        
        # 2. Анализ плеч (разные критерии для стоя/сидя)
        if posture_type == PostureType.STANDING:
            shoulder_angle = calculate_angle(ls[:2], (ls+rs)[:2]/2, rs[:2])
            if shoulder_angle < 160:  # Округление плеч
                messages.append("• Плечи ссутулены")
        else:
            if abs(ls[0] - rs[0]) < 0.15*image.shape[1]:  # Плечи сведены
                messages.append("• Плечи сведены вперед")
        
        # 3. Анализ позвоночника
        if lh[2] > 0.3 and rh[2] > 0.3:
            spine_angle = calculate_angle((ls+rs)[:2]/2, (lh+rh)[:2]/2, 
                                        (lh+rh)[:2]/2 + np.array([0, 100]))
            if spine_angle < 75 or spine_angle > 105:
                messages.append("• Искривление позвоночника")
        
        # Формируем отчет с учетом типа позы
        if messages:
            report = [
                f"**Поза {posture_type.value} - проблемы:**",
                *messages,
                "\n**Рекомендации:**",
                "• Держите голову прямо" if head_forward else "",
                "• Расправьте плечи" if "Плечи" in "\n".join(messages) else "",
                "• Выпрямите спину" if "позвоночника" in "\n".join(messages) else ""
            ]
            report = [r for r in report if r]  # Удаляем пустые строки
        else:
            report = [f"**Отличная осанка ({posture_type.value})!**"]
        
        return annotated_image, "\n".join(report), posture_type
        
    except Exception as e:
        return image, f"Ошибка анализа: {str(e)}", PostureType.UNKNOWN

def main():
    st.set_page_config(layout="wide")
    st.title("📷 Умный анализатор осанки")
    
    # Инициализация состояния
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False
    
    # Кнопка включения камеры
    if not st.session_state.camera_on:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Включить камеру", type="primary"):
                st.session_state.camera_on = True
                st.rerun()
        with col2:
            st.info("Для лучшего анализа встаньте/сядьте так, чтобы было видно плечи и бедра")
        return
    
    # Основной интерфейс
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Видео с камеры")
        frame_placeholder = st.empty()
    
    with col2:
        st.header("Анализ осанки")
        status_placeholder = st.empty()
        posture_placeholder = st.empty()
        
        if st.button("Выключить камеру"):
            st.session_state.camera_on = False
            st.rerun()
            return
    
    # Работа с камерой
    cap = cv2.VideoCapture(0)
    try:
        while st.session_state.camera_on:
            ret, frame = cap.read()
            if not ret:
                st.error("Ошибка подключения к камере")
                break
            
            # Анализ и отображение
            analyzed, status, posture = analyze_posture(frame)
            frame_placeholder.image(analyzed, channels="BGR")
            status_placeholder.markdown(status)
            posture_placeholder.markdown(f"**Текущая поза:** {posture.value}")
            
            time.sleep(0.1)
    finally:
        cap.release()

if __name__ == "__main__":
    main()