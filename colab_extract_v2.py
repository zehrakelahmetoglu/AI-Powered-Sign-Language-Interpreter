import os
import cv2
import json
import numpy as np
import pandas as pd
import mediapipe as mp_lib
from tqdm import tqdm
from pathlib import Path
import argparse

# Ayarlar
TARGET_FRAMES = 64
INPUT_DIM = 258

class RobustExtractor:
    def __init__(self, model_complexity=1):
        self.holistic = mp_lib.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_normalized(self, results):
        """Kişiyi 0-1 arasına sığdıran (Bounding Box) normalizasyon."""
        vec = np.zeros(INPUT_DIM, dtype=np.float32)
        points_x, points_y = [], []
        pose_lms, lh_lms, rh_lms = [], [], []

        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                pose_lms.append([lm.x, lm.y, lm.z, lm.visibility])
                points_x.append(lm.x); points_y.append(lm.y)
        else: pose_lms = [[0,0,0,0]] * 33

        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                lh_lms.append([lm.x, lm.y, lm.z])
                points_x.append(lm.x); points_y.append(lm.y)
        else: lh_lms = [[0,0,0]] * 21

        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                rh_lms.append([lm.x, lm.y, lm.z])
                points_x.append(lm.x); points_y.append(lm.y)
        else: rh_lms = [[0,0,0]] * 21

        if points_x and points_y:
            min_x, max_x = min(points_x), max(points_x)
            min_y, max_y = min(points_y), max(points_y)
            w = (max_x - min_x) + 1e-6
            h = (max_y - min_y) + 1e-6
            
            for i, (x, y, z, v) in enumerate(pose_lms):
                vec[i*4 : i*4+4] = [(x - min_x)/w, (y - min_y)/h, z, v]
            for i, (x, y, z) in enumerate(lh_lms):
                vec[132 + i*3 : 132 + i*3+3] = [(x - min_x)/w, (y - min_y)/h, z]
            for i, (x, y, z) in enumerate(rh_lms):
                vec[195 + i*3 : 195 + i*3+3] = [(x - min_x)/w, (y - min_y)/h, z]
        return vec

    def process_video(self, video_path, out_path):
        cap = cv2.VideoCapture(str(video_path))
        frames_raw = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames_raw.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames_raw: return False
        n = len(frames_raw)
        indices = np.linspace(0, n - 1, TARGET_FRAMES, dtype=int)
        keypoints = np.zeros((TARGET_FRAMES, INPUT_DIM), dtype=np.float32)
        for t, idx in enumerate(indices):
            results = self.holistic.process(frames_raw[idx])
            keypoints[t] = self.extract_normalized(results)
        np.save(out_path, keypoints)
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--out_dir", default="keypoints_v2")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv_path, header=None, names=["id", "label"])
    extractor = RobustExtractor()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        v_path = Path(args.video_dir) / f"{row['id']}_color.mp4"
        o_path = Path(args.out_dir) / f"{row['id']}.npy"
        if not o_path.exists() and v_path.exists():
            extractor.process_video(v_path, o_path)

if __name__ == "__main__":
    main()
