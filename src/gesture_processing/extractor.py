import cv2
import mediapipe as mp
import numpy as np
import json
from typing import Dict, Optional

class GestureLandmarkExtractor:
    def __init__(self,
                 frame_interval_ms: int = 100,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.frame_interval_ms = frame_interval_ms
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def extract(self, video_path: str, output_json: str, gesture_label: str) -> Optional[str]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Не удалось открыть {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = max(1, int((self.frame_interval_ms / 1000) * fps))
        print(f"[INFO] Обработка {video_path}, FPS: {fps}, шаг: {frame_interval}")

        landmarks_data = {
            "video": video_path,
            "gesture": gesture_label,
            "fps": fps,
            "frames": []
        }

        frame_idx, saved = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                lm = self._process_frame(frame)
                if lm:
                    landmarks_data["frames"].append({
                        "frame_idx": frame_idx,
                        **lm
                    })
                    saved += 1
            frame_idx += 1

        cap.release()

        if landmarks_data["frames"]:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(landmarks_data, f, ensure_ascii=False, indent=2)
            print(f"[OK] {gesture_label}: сохранено {saved} кадров → {output_json}")
            return output_json
        else:
            print(f"[WARN] Пустые данные: {video_path}")
            return None

    def _process_frame(self, frame) -> Dict:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.holistic.process(img)

        def to_list(lm_list):
            return [
                {"x": lm.x, "y": lm.y, "z": lm.z,
                 **({"visibility": lm.visibility} if hasattr(lm, "visibility") else {})}
                for lm in lm_list
            ]

        return {
            "pose": to_list(res.pose_landmarks.landmark) if res.pose_landmarks else [],
            "face": to_list(res.face_landmarks.landmark) if res.face_landmarks else [],
            "left_hand": to_list(res.left_hand_landmarks.landmark) if res.left_hand_landmarks else [],
            "right_hand": to_list(res.right_hand_landmarks.landmark) if res.right_hand_landmarks else []
        }
