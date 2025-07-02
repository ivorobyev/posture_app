import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import numpy as np
import mediapipe as mp
import queue
import threading
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseAnalyzer(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.status_queue = queue.Queue(maxsize=1)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        results = self.pose.process(img)

        annotated_img = img.copy()

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )
            status = self.analyze_posture(results.pose_landmarks.landmark)
        else:
            status = "Ключевые точки не обнаружены"

        # Положить последний статус в очередь, перезаписывая старый
        try:
            self.status_queue.get_nowait()
        except queue.Empty:
            pass
        self.status_queue.put(status)

        return av.VideoFrame.from_ndarray(annotated_img, format="rgb24")

    def analyze_posture(self, landmarks):
        # Пример простого анализа с ключевыми точками
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = landmarks[mp_pose.PoseLandmark.NOSE]

        messages = []
        if nose.y > (left_shoulder.y + right_shoulder.y)/2 + 0.1:
            messages.append("Голова наклонена вперед")

        if abs(left_shoulder.y - right_shoulder.y) > 0.05:
            messages.append("Асимметрия плеч")

        if messages:
            return "Проблемы с осанкой:\n" + "\n".join(messages)
        else:
            return "Осанка в норме"

def main():
    st.title("Анализатор осанки")

    webrtc_ctx = webrtc_streamer(
        key="pose-analyzer",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=PoseAnalyzer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    status_placeholder = st.empty()

    # Запускаем обновление текста в отдельном потоке
    def update_status():
        while True:
            if webrtc_ctx.video_processor:
                try:
                    status = webrtc_ctx.video_processor.status_queue.get(timeout=1)
                    status_placeholder.markdown(f"**Статус:**\n\n{status}")
                except queue.Empty:
                    pass
            time.sleep(1)
            st.experimental_rerun()

    if webrtc_ctx.state.playing:
        threading.Thread(target=update_status, daemon=True).start()

if __name__ == "__main__":
    main()