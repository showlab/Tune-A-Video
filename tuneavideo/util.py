import os
import imageio
import numpy as np

import torch
import torchvision

from einops import rearrange


def save_videos_grid(
    videos: torch.Tensor, 
    save_path: str = 'output',
    path: str = 'output.gif', 
    rescale=False, 
    n_rows=4, 
    fps=3
):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    imageio.mimsave(os.path.join(save_path, path), outputs, fps=fps)