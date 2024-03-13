import os
import pathlib

import mediapipe as mp
import numpy as np
import torchvision


class LandmarksDetector:
    def __init__(self):
        self.face_detector = mp.tasks.vision.FaceDetector

        self.options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=os.path.join(
                    os.path.dirname(pathlib.Path(__file__).parent.parent.absolute()),
                    "models",
                    "mediapipe",
                    "short_range.tflite",
                )
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )

    def __call__(self, filename):
        video_frames = torchvision.io.read_video(filename, pts_unit="sec")[0].numpy()

        with self.face_detector.create_from_options(self.options) as detector:
            landmarks = []
            for frame in video_frames:
                results = detector.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                )

                if not results.detections:
                    landmarks.append(None)
                    continue

                face_points = []
                for idx, detected_faces in enumerate(results.detections):
                    max_id, max_size = 0, 0

                    bbox = detected_faces.bounding_box
                    bbox_size = (bbox.width - bbox.origin_x) + (
                        bbox.height - bbox.origin_y
                    )
                    if bbox_size > max_size:
                        max_id, max_size = idx, bbox_size

                    keypoints = detected_faces.keypoints
                    lmx = [
                        [
                            int(keypoints[0].x * frame.shape[0]),
                            int(keypoints[0].y * frame.shape[1]),
                        ],
                        [
                            int(keypoints[1].x * frame.shape[0]),
                            int(keypoints[1].y * frame.shape[1]),
                        ],
                        [
                            int(keypoints[2].x * frame.shape[0]),
                            int(keypoints[2].y * frame.shape[1]),
                        ],
                        [
                            int(keypoints[3].x * frame.shape[0]),
                            int(keypoints[3].y * frame.shape[1]),
                        ],
                    ]

                    face_points.append(lmx)
                landmarks.append(np.array(face_points[max_id]))
            return landmarks
