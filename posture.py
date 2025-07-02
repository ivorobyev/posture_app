import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode

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
    if len(point) >= 3 and point[2] > 0.25:
        return point[:3]
    return fallback if fallback is not None else np.array([0, 0, 0])

def analyze_posture(image):
    """Анализ осанки для сидячего положения"""
    try:
        results = model(image, verbose=False)
        
        if not results or len(results) == 0:
            return image, "Повернитесь к камере лицом"
            
        annotated_image = results[0].plot()
        keypoints = results[0].keypoints.xy[0].cpu().numpy() if results[0].keypoints else None
        
        if keypoints is None or len(keypoints) < 6:  # Минимум: голова и плечи
            return annotated_image, "Встаньте так, чтобы были видны плечи"
        
        # Получаем ключевые точки верхней части тела
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        LEFT_EAR = 3
        RIGHT_EAR = 4
        NOSE = 0
        
        ls = get_keypoint_safe(keypoints, LEFT_SHOULDER)
        rs = get_keypoint_safe(keypoints, RIGHT_SHOULDER)
        le = get_keypoint_safe(keypoints, LEFT_EAR, ls)
        re = get_keypoint_safe(keypoints, RIGHT_EAR, rs)
        nose = get_keypoint_safe(keypoints, NOSE, (ls + rs) / 2)
        
        # Анализ осанки
        messages = []
        
        # 1. Наклон головы вперед
        head_forward = (nose[1] > (ls[1] + rs[1])/2 + 0.1*image.shape[0])
        if head_forward:
            messages.append("• Голова наклонена вперед (текстовая шея)")
        
        # 2. Сутулость плеч
        shoulders_rounded = (ls[0] > rs[0] + 0.1*image.shape[1]) or \
                          (rs[0] < ls[0] - 0.1*image.shape[1])
        if shoulders_rounded:
            messages.append("• Плечи ссутулены")
        
        # Формируем отчет
        if messages:
            report = [
                "**Проблемы с осанкой (сидя):**",
                *messages,
                "\n**Рекомендации:**",
                "• Поднимите экран на уровень глаз",
                "• Обопритесь на спинку кресла",
                "• Следите за положением головы"
            ]
        else:
            report = ["**Отличная осанка!**", "Вы правильно сидите за столом"]
        
        return annotated_image, "\n".join(report)
        
    except Exception as e:
        return image, f"Ошибка анализа: {str(e)}"

def video_frame_callback(frame):
    """Обработка кадров для WebRTC"""
    img = frame.to_ndarray(format="bgr24")
    analyzed_img, posture_status = analyze_posture(img)
    return av.VideoFrame.from_ndarray(analyzed_img, format="bgr24")

def main():
    st.set_page_config(layout="wide")
    st.title("📷 Анализатор осанки для офиса")
    
    st.markdown("""
    ## Инструкция:
    1. Сядьте за рабочий стол как обычно
    2. Разрешите доступ к камере
    3. Расположитесь так, чтобы были видны плечи и голова
    """)
    
    # WebRTC поток для работы в браузере
    webrtc_ctx = webrtc_streamer(
        key="posture",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    if not webrtc_ctx.state.playing:
        st.warning("Ожидание доступа к камере...")

if __name__ == "__main__":
    main()