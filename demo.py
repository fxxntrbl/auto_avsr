import os

import hydra
import torch
import torchvision

from preprocessing import ModelModule
from preprocessing.data import VideoTransform
from preprocessing.detector import LandmarksDetector, VideoProcess


class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg):
        super(InferencePipeline, self).__init__()

        self.landmarks_detector = LandmarksDetector()
        self.video_process = VideoProcess(convert_gray=False)

        self.video_transform = VideoTransform(subset="test")

        self.modelmodule = ModelModule(cfg)
        self.modelmodule.model.load_state_dict(
            torch.load(
                "models/visual/model.pth",
                map_location=lambda storage, loc: storage,
            )
        )
        self.modelmodule.eval()

    def forward(self, filename):
        filename = os.path.abspath(filename)
        assert os.path.isfile(filename), f"filename: {filename} does not exist."

        video = self.load_video(filename)

        with torch.no_grad():
            transcript = self.modelmodule(video)

        return transcript

    def load_video(self, filename):
        video = torchvision.io.read_video(filename, pts_unit="sec")[0].numpy()
        landmarks = self.landmarks_detector(video)
        video = self.video_process(video, landmarks)
        video = torch.tensor(video)
        video = video.permute((0, 3, 1, 2))
        video = self.video_transform(video)
        return video


@hydra.main(version_base="1.3", config_path="configs", config_name="hydra")
def main(cfg):
    pipeline = InferencePipeline(cfg)
    transcript = pipeline(cfg.filename)
    print(f"transcript: {transcript}")


if __name__ == "__main__":
    main()
