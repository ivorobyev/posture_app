import streamlit as st
import av
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import mediapipe as mp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class MediaPipePostureAnalyzer(VideoProcessorBase):
    def __init__(self):
        self.status = "Ожидание анализа..."
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.result_frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            results = self.pose.process(img_rgb)
            if not results.pose_landmarks:
                self.status = "🙈 Лицо/плечи не видны"
                self.result_frame = img
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            # Рисуем аннотации
            annotated_img = img.copy()
            mp_drawing.draw_landmarks(
                annotated_img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )

            # Извлечение ключевых точек для анализа
            lm = results.pose_landmarks.landmark
            h, w, _ = img.shape

            def get_point(name):
                point = getattr(mp_pose.PoseLandmark, name)
                lm_point = lm[point]
                if lm_point.visibility < 0.5:
                    return None
                return np.array([lm_point.x * w, lm_point.y * h])

            nose = get_point("NOSE")
            left_shoulder = get_point("LEFT_SHOULDER")
            right_shoulder = get_point("RIGHT_SHOULDER")

            messages = []

            if nose is not None and left_shoulder is not None and right_shoulder is not None:
                avg_sh_y = (left_shoulder[1] + right_shoulder[1]) / 2
                head_tilt_threshold = 20  # в пикселях, можно подстраивать

                if nose[1] > avg_sh_y + head_tilt_threshold:
                    messages.append("• Голова наклонена вперёд")

                shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
                shoulder_threshold = 15  # в пикселях

                if shoulder_diff > shoulder_threshold:
                    messages.append("• Асимметрия плеч")

            self.status = (
                "**Осанка хорошая!**\nВы сидите правильно."
                if not messages
                else "**Проблемы с осанкой:**\n" + "\n".join(messages)
            )

            self.result_frame = annotated_img
            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

        except Exception as e:
            logger.error(f"Ошибка анализа: {e}")
            self.status = "Ошибка анализа"
            return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.set_page_config(layout="wide")
    st.title("🧍 Анализ осанки с MediaPipe и камеры")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Видео с камеры")
        webrtc_ctx = webrtc_streamer(
            key="mediapipe-pose-stream",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=MediaPipePostureAnalyzer,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col2:
        st.header("Анализ осанки")
        info_box = st.empty()

        if webrtc_ctx.video_processor:
            status = webrtc_ctx.video_processor.status
            info_box.markdown(f"""
                <div style='background:#f0f2f6;padding:20px;border-radius:10px; font-family: monospace; white-space: pre-line;'>
                {status}
                </div>
            """, unsafe_allow_html=True)

        st.subheader("Советы для здоровой осанки:")
        st.markdown("""
        - Монитор на уровне глаз  
        - Спина ровная  
        - Плечи расслаблены  
        - Делайте перерывы каждый час
        """)

if __name__ == "__main__":
    main()