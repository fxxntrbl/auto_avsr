from retinaface import RetinaFace
import numpy as np


class LandmarksDetector:
    def __init__(self) -> None:
        self.detector = RetinaFace.detect_faces

    def __call__(self, video_frames):
        total_landmarks = []

        for frame in video_frames:
            detected_faces = self.detector(frame, threshold=0.8)
            max_id, max_size = (
                0,
                0,
            )  # we should edit here later for multi-party conversations

            face_points = []
            new_array = [value for key, value in detected_faces.items()]

            for idx, result in enumerate(new_array):
                current_bbox = result["facial_area"]
                current_landmarks = result["landmarks"]

                bbox_size = (current_bbox[2] - current_bbox[0]) + (
                    current_bbox[3] - current_bbox[1]
                )
                if bbox_size > max_size:
                    max_id, max_size = idx, bbox_size

                face_points.append(
                    [
                        current_landmarks["right_eye"],
                        current_landmarks["left_eye"],
                        current_landmarks["nose"],
                        [
                            (
                                current_landmarks["mouth_left"][0]
                                + current_landmarks["mouth_right"][0]
                            )
                            / 2,
                            (
                                current_landmarks["mouth_left"][1]
                                + current_landmarks["mouth_right"][1]
                            )
                            / 2,
                        ],
                    ]
                )
            total_landmarks.append(np.array(face_points[max_id]))
        return total_landmarks
