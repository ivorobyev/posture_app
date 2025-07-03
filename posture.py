import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import mediapipe as mp
import av
from collections import deque

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5,
                                min_tracking_confidence=0.5,
                                model_complexity=1)
        self.status_deque = deque(maxlen=1)

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
            messages.append("• Head tilted forward (text neck) (頭が前に傾いている（テキストネック）)")

        shoulders_rounded = left_shoulder.x > left_hip.x + 0.05 or right_shoulder.x < right_hip.x - 0.05
        if shoulders_rounded:
            messages.append("• Rounded shoulders (肩が丸まっている)")

        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        hip_diff = abs(left_hip.y - right_hip.y)
        if shoulder_diff > 0.05 or hip_diff > 0.05:
            messages.append("• Side tilt (asymmetrical posture) (体が傾いている（非対称な姿勢）)")

        if sitting and (left_hip.y < left_shoulder.y + 0.15 or right_hip.y < right_shoulder.y + 0.15):
            messages.append("• Pelvis tilted forward (while sitting) (骨盤が前に傾いている（座っている時）)")

        if messages:
            report = [
                f"**{'Sitting' if sitting else 'Standing'} - posture issues detected ({'座っている' if sitting else '立っている'} - 姿勢の問題が検出されました):**",
                *messages,
                "\n**Recommendations (アドバイス):**",
                "• Keep your head straight, ears should be over shoulders (頭をまっすぐに保ち、耳は肩の上に来るように)",
                "• Pull your shoulders back and down (肩を後ろに引き下げる)",
                "• Keep your back straight, avoid side tilts (背中をまっすぐに保ち、横に傾かないように)",
                "• When sitting, rest on your sitting bones (座っている時は坐骨で支える)"
            ]
        else:
            report = [
                f"**Excellent posture ({'sitting' if sitting else 'standing'})! (素晴らしい姿勢（{'座っている' if sitting else '立っている'}）!)**",
                "All key points are in correct position. (すべてのキーポイントが正しい位置にあります)",
                "\n**Tip (ヒント):**",
                "• Continue to monitor your posture throughout the day (一日中姿勢に気を配り続けましょう)"
            ]

        return "\n\n".join(report)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.pose.process(img_rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            status = self.analyze_posture(results.pose_landmarks, img.shape)
            self.status_deque.append(status)
        else:
            self.status_deque.append("Key points not detected (キーポイントが検出されませんでした)")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.set_page_config(layout="wide")
    st.title("📷 Posture Analyzer with Web Camera (WebRTC) (ウェブカメラを使った姿勢分析)")
    st.write("This app analyzes your posture in real time using your browser's camera. (このアプリはブラウザのカメラを使ってリアルタイムで姿勢を分析します)")

    webrtc_ctx = webrtc_streamer(
        key="pose-analyzer",
        video_processor_factory=PoseProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    if webrtc_ctx.video_processor:
        if webrtc_ctx.video_processor.status_deque:
            analysis_text = webrtc_ctx.video_processor.status_deque[-1]
        else:
            analysis_text = "Waiting for video and analysis... (動画と分析を待っています...)"

        st.markdown("### Posture Analysis (姿勢分析):")
        st.markdown(analysis_text)

if __name__ == "__main__":
    main()