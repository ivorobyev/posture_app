import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
from enum import Enum

class PostureType(Enum):
    SITTING = "сидя"
    STANDING = "стоя"
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
    if len(point) >= 3 and point[2] > 0.25:  # Пониженный порог для сложных ракурсов
        return point[:3]
    return fallback if fallback is not None else np.array([0, 0, 0])

def calculate_angle(a, b, c):
    """Вычисление угла между тремя точками"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1, 1)))

def determine_posture_type(keypoints, img_height):
    """Определение позы с приоритетом для сидячего положения"""
    # Приоритетная проверка для сидячего положения
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12
    
    ls = get_keypoint_safe(keypoints, LEFT_SHOULDER)
    rs = get_keypoint_safe(keypoints, RIGHT_SHOULDER)
    lh = get_keypoint_safe(keypoints, LEFT_HIP)
    rh = get_keypoint_safe(keypoints, RIGHT_HIP)
    
    # Критерии для сидячего положения (когда бедра не видны)
    if lh[2] < 0.3 or rh[2] < 0.3:
        # Если бедра не видны, считаем что пользователь сидит (основной сценарий)
        return PostureType.SITTING
    
    # Дополнительные проверки, если бедра видны
    shoulder_hip_ratio = ((ls[1] - lh[1]) + (rs[1] - rh[1])) / (2 * img_height)
    if shoulder_hip_ratio < 0.15:  # Плечи близко к бедрам
        return PostureType.SITTING
    
    return PostureType.STANDING

def analyze_sitting_posture(keypoints, img_size):
    """Специализированный анализ для сидячего положения"""
    NOSE = 0
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    
    img_h, img_w = img_size
    
    # Получаем точки верхней части тела
    ls = get_keypoint_safe(keypoints, LEFT_SHOULDER)
    rs = get_keypoint_safe(keypoints, RIGHT_SHOULDER)
    le = get_keypoint_safe(keypoints, LEFT_EAR, ls)
    re = get_keypoint_safe(keypoints, RIGHT_EAR, rs)
    nose = get_keypoint_safe(keypoints, NOSE, (ls + rs) / 2)
    lelb = get_keypoint_safe(keypoints, LEFT_ELBOW)
    relb = get_keypoint_safe(keypoints, RIGHT_ELBOW)
    
    messages = []
    
    # 1. Анализ головы (основная проблема при сидении)
    head_forward = False
    if le[2] > 0.25 and re[2] > 0.25:  # Если видны оба уха
        head_angle = calculate_angle(le[:2], nose[:2], re[:2])
        if head_angle < 150:  # Наклон головы вперед
            head_forward = True
    elif nose[2] > 0.3:
        if nose[1] > (ls[1] + rs[1])/2 + 0.12*img_h:  # Нос значительно ниже плеч
            head_forward = True
    
    if head_forward:
        messages.append("• Голова наклонена вперед (текстовая шея)")
    
    # 2. Анализ плеч и локтей (для выявления сутулости)
    if lelb[2] > 0.3 and relb[2] > 0.3:
        # Проверка, если локти перед плечами (сутулость)
        if (lelb[0] > ls[0] + 0.05*img_w) or (relb[0] < rs[0] - 0.05*img_w):
            messages.append("• Плечи ссутулены (локти выдвинуты вперед)")
    else:
        # Альтернативная проверка по положению плеч
        shoulder_angle = calculate_angle(ls[:2], (ls+rs)[:2]/2, rs[:2])
        if shoulder_angle < 160:  # Округление плеч
            messages.append("• Плечи сведены вперед")
    
    # 3. Анализ асимметрии (сколиоз)
    if ls[2] > 0.3 and rs[2] > 0.3:
        shoulder_diff = abs(ls[1] - rs[1]) / img_h
        if shoulder_diff > 0.08:  # Разница высоты плеч
            messages.append("• Асимметрия плеч (возможен сколиоз)")
    
    return messages

def analyze_posture(image):
    """Анализ осанки с упором на сидячее положение"""
    try:
        results = model(image, verbose=False)
        
        if not results or len(results) == 0 or results[0].keypoints is None:
            return image, "Встаньте перед камерой", PostureType.UNKNOWN
        
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        annotated_image = results[0].plot()
        
        if len(keypoints) < 13:  # Минимум: голова, плечи, локти
            return annotated_image, "Повернитесь к камере верхней частью тела", PostureType.UNKNOWN
        
        # Определяем позу (с приоритетом сидячего положения)
        posture_type = determine_posture_type(keypoints, image.shape[0])
        
        # Специализированный анализ для сидячего положения
        if posture_type == PostureType.SITTING:
            messages = analyze_sitting_posture(keypoints, image.shape[:2])
            
            if messages:
                report = [
                    "**Проблемы с осанкой (сидя):**",
                    *messages,
                    "\n**Рекомендации:**",
                    "• Поднимите экран на уровень глаз",
                    "• Обопритесь на спинку кресла",
                    "• Поставьте ноги на пол"
                ]
            else:
                report = ["**Отличная осанка!**", "Вы правильно сидите за столом"]
        else:
            report = ["**Вы стоите**", "Для анализа осанки сядьте за стол"]
        
        return annotated_image, "\n".join(report), posture_type
        
    except Exception as e:
        return image, f"Ошибка анализа: {str(e)}", PostureType.UNKNOWN

def main():
    st.set_page_config(layout="wide")
    st.title("📷 Анализатор осанки для офисных работников")
    
    # Инициализация состояния
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False
    
    # Кнопка включения камеры
    if not st.session_state.camera_on:
        st.markdown("""
        ## Инструкция:
        1. Сядьте за рабочий стол как обычно
        2. Включите камеру
        3. Расположитесь так, чтобы было видно ваши плечи и голову
        """)
        
        if st.button("Включить камеру", type="primary"):
            st.session_state.camera_on = True
            st.rerun()
        return
    
    # Основной интерфейс
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Ваша поза")
        frame_placeholder = st.empty()
    
    with col2:
        st.header("Анализ осанки")
        status_placeholder = st.empty()
        
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
            analyzed, status, _ = analyze_posture(frame)
            frame_placeholder.image(analyzed, channels="BGR")
            status_placeholder.markdown(status)
            
            time.sleep(0.1)
    finally:
        cap.release()

if __name__ == "__main__":
    main()