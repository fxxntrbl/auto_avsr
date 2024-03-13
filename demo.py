import os

import hydra
import torch
import torchaudio
import torchvision

from datamodule.av_dataset import cut_or_pad
from datamodule.transforms import AudioTransform, VideoTransform
from lightning_av import ModelModule
from preparation.detectors.detector import LandmarksDetector
from preparation.detectors.video_process import VideoProcess


class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg):
        super(InferencePipeline, self).__init__()
        self.modality = cfg.modality
        self.audio_transform = AudioTransform(subset="test")

        self.landmarks_detector = LandmarksDetector()
        self.video_process = VideoProcess(convert_gray=False)

        self.video_transform = VideoTransform(subset="test")

        self.modelmodule = ModelModule(cfg)
        self.modelmodule.model.load_state_dict(
            torch.load(
                cfg.pretrained_model_path, map_location=lambda storage, loc: storage
            )
        )
        self.modelmodule.eval()

    def forward(self, data_filename):
        data_filename = os.path.abspath(data_filename)
        assert os.path.isfile(
            data_filename
        ), f"data_filename: {data_filename} does not exist."

        if self.modality in ["audio", "audiovisual"]:
            audio, sample_rate = self.load_audio(data_filename)
            audio = self.audio_process(audio, sample_rate)
            audio = audio.transpose(1, 0)
            audio = self.audio_transform(audio)

        if self.modality in ["video", "audiovisual"]:
            video = self.load_video(data_filename)
            landmarks = self.landmarks_detector(video)
            video = self.video_process(video, landmarks)
            video = torch.tensor(video)
            video = video.permute((0, 3, 1, 2))
            video = self.video_transform(video)

        if self.modality == "audiovisual":
            print(len(audio), len(video))
            assert (
                530 < len(audio) // len(video) < 670
            ), "The video frame rate should be between 24 and 30 fps."

            rate_ratio = len(audio) // len(video)
            if rate_ratio == 640:
                pass
            else:
                print(
                    f"The ideal video frame rate is set to 25 fps, but the current frame rate ratio, calculated as {len(video)*16000/len(audio):.1f}, which may affect the performance."
                )
                audio = cut_or_pad(audio, len(video) * 640)
            with torch.no_grad():
                transcript = self.modelmodule(video, audio)

        return transcript

    def load_audio(self, data_filename):
        waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        return waveform, sample_rate

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform


@hydra.main(version_base="1.3", config_path="configs", config_name="hydra")
def main(cfg):
    pipeline = InferencePipeline(cfg)
    transcript = pipeline(cfg.demo)
    print(f"transcript: {transcript}")


if __name__ == "__main__":
    main()
