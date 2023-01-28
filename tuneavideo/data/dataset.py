import decord
decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange


class TuneAVideoDataset(Dataset):
    def __init__(
            self,
            video_path: str,
            prompt: str,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
    ):
        self.video_path = video_path
        self.prompt = prompt
        self.prompt_ids = None

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # load and sample video frames
        vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)
        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids
        }

        return example
