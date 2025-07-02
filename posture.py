import streamlit as st
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import mediapipe as mp
from PIL import Image, ImageDraw

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class MediaPipePostureAnalyzer(VideoProcessorBase):
    def __init__(self):
        self.status = "Ожидание анализа..."
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def draw_landmarks(self, image, landmarks, connections):
        # Рисуем линии и точки на PIL изображении
        draw = ImageDraw.Draw(image)
        w, h = image.size

        # Рисуем линии соединений
        for connection in connections:
            start_idx, end_idx = connection
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            if start.visibility < 0.5 or end.visibility < 0.5:
                continue
            draw.line(
                [(start.x * w, start.y * h), (end.x * w, end.y * h)],
                fill=(0, 255, 0), width=3
            )
        # Рисуем точки
        for lm in landmarks:
            if lm.visibility < 0.5:
                continue
            x, y = lm.x * w, lm.y * h
            r = 5
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        img_pil = Image.fromarray(img)

        results = self.pose.process(img)
        if not results.pose_landmarks:
            self.status = "🙈 Лицо/плечи не видны"
            return av.VideoFrame.from_ndarray(np.array(img_pil), format="rgb24")

        self.draw_landmarks(img_pil, results.pose_landmarks.landmark, mp_pose.POSE_CONNECTIONS)

        # Анализ осанки
        lm = results.pose_landmarks.landmark
        w, h = img_pil.size

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

        self.status = (
            "**Осанка хорошая!**\nВы сидите правильно."
            if not messages
            else "**Проблемы с осанкой:**\n" + "\n".join(messages)
        )

        return av.VideoFrame.from_ndarray(np.array(img_pil), format="rgb24")


def main():
    st.set_page_config(layout="wide")
    st.title("🧍 Анализ осанки с MediaPipe без OpenCV")

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