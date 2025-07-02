import streamlit as st
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class MediaPipePostureAnalyzer(VideoProcessorBase):
    def __init__(self):
        self.status = "Ожидание анализа..."
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2),
            )

            lm = results.pose_landmarks.landmark
            h, w, _ = img.shape

            def get_point(name):
                pt = getattr(mp_pose.PoseLandmark, name)
                lm_point = lm[pt]
                return None if lm_point.visibility < 0.5 else (lm_point.x * w, lm_point.y * h)

            nose = get_point("NOSE")
            left_shoulder = get_point("LEFT_SHOULDER")
            right_shoulder = get_point("RIGHT_SHOULDER")

            messages = []
            if nose and left_shoulder and right_shoulder:
                avg_sh_y = (left_shoulder[1] + right_shoulder[1]) / 2
                head_tilt_threshold = 20  # pixels

                if nose[1] > avg_sh_y + head_tilt_threshold:
                    messages.append("• Голова наклонена вперёд")

                shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
                shoulder_threshold = 15  # pixels

                if shoulder_diff > shoulder_threshold:
                    messages.append("• Асимметрия плеч")

            if messages:
                self.status = "**Проблемы с осанкой:**\n" + "\n".join(messages)
            else:
                self.status = "**Осанка хорошая!**\nВы сидите правильно."
        else:
            self.status = "🙈 Лицо/плечи не видны"

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.set_page_config(layout="wide")
    st.title("🧍 Анализ осанки с MediaPipe и OpenCV")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Видео с камеры")
        webrtc_ctx = webrtc_streamer(
            key="mediapipe-opencv-pose-stream",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=MediaPipePostureAnalyzer,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col2:
        st.header("Анализ осанки")
        info_box = st.empty()
        if webrtc_ctx.video_processor:
            info_box.markdown(f"""
                <div style='background:#f0f2f6;padding:20px;border-radius:10px; white-space: pre-line; font-family: monospace;'>
                {webrtc_ctx.video_processor.status}
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