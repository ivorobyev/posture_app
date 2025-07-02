import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import mediapipe as mp
import av
from collections import deque

# Инициализация MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5,
                                min_tracking_confidence=0.5,
                                model_complexity=1)
        self.status_deque = deque(maxlen=1)  # Для передачи текста в UI

    def analyze_posture(self, landmarks, image_shape):
        h, w, _ = image_shape
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        left_ear = landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
        nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]

        sitting = left_hip.y < left_shoulder.y + 0.1 or right_hip.y < right_shoulder.y + 0.1

        messages = []

        head_forward = (left_ear.y > left_shoulder.y + 0.1 or right_ear.y > right_shoulder.y + 0.1) and \
                       (nose.y > left_shoulder.y or nose.y > right_shoulder.y)
        if head_forward:
            messages.append("• Голова наклонена вперед (текстовая шея)")

        shoulders_rounded = left_shoulder.x > left_hip.x + 0.05 or right_shoulder.x < right_hip.x - 0.05
        if shoulders_rounded:
            messages.append("• Плечи ссутулены (округлены вперед)")

        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        hip_diff = abs(left_hip.y - right_hip.y)
        if shoulder_diff > 0.05 or hip_diff > 0.05:
            messages.append("• Наклон в сторону (несимметричная осанка)")

        if sitting and (left_hip.y < left_shoulder.y + 0.15 or right_hip.y < right_shoulder.y + 0.15):
            messages.append("• Таз наклонен вперед (сидя)")

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

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.pose.process(img_rgb)
        if results.pose_landmarks:
            # Рисуем ключевые точки
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            status = self.analyze_posture(results.pose_landmarks, img.shape)
            self.status_deque.append(status)
        else:
            self.status_deque.append("Ключевые точки не обнаружены")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.set_page_config(layout="wide")
    st.title("📷 Анализатор осанки с веб-камеры (WebRTC)")
    st.write("Приложение анализирует вашу осанку в реальном времени с помощью камеры браузера.")

    # Запускаем WebRTC стример с нашим процессором
    webrtc_ctx = webrtc_streamer(
        key="pose-analyzer",
        video_processor_factory=PoseProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    # Выводим текст анализа из процесса
    if webrtc_ctx.video_processor:
        # Получаем последнюю доступную строку с анализом
        if webrtc_ctx.video_processor.status_deque:
            analysis_text = webrtc_ctx.video_processor.status_deque[-1]
        else:
            analysis_text = "Ожидание видео и анализа..."

        st.markdown("### Анализ осанки:")
        st.markdown(analysis_text)

if __name__ == "__main__":
    main()