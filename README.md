# Tune-A-Video

This repository is the official implementation of [Tune-A-Video](https://arxiv.org/abs/2212.11565).

**[Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation](https://arxiv.org/abs/2212.11565)**
<br/>
[Jay Zhangjie Wu](https://zhangjiewu.github.io/), 
[Yixiao Ge](https://geyixiao.com/), 
[Xintao Wang](https://xinntao.github.io/), 
[Stan Weixian Lei](), 
[Yuchao Gu](https://ycgu.site/), 
[Wynne Hsu](https://www.comp.nus.edu.sg/~whsu/), 
[Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en), 
[Xiaohu Qie](https://scholar.google.com/citations?user=mk-F69UAAAAJ&hl=en), 
[Mike Zheng Shou](https://sites.google.com/view/showlab)
<br/>

[Project Page](https://tuneavideo.github.io/) | [arXiv](https://arxiv.org/abs/2212.11565)

## Setup

### Requirements

```shell
pip install -r requirements.txt
```

Installing [xformers](https://github.com/facebookresearch/xformers) is highly recommended for more efficiency and speed on GPUs. 
To enable xformers, set `enable_xformers_memory_efficient_attention=True` (default).

### Weights

You can download the pre-trained [Stable Diffusion](https://arxiv.org/abs/2112.10752) models 
(e.g., [Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)):

```shell
git lfs install
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
```

Alternatively, you can use a personalized [DreamBooth](https://arxiv.org/abs/2208.12242) model (e.g., [mr-potato-head](https://huggingface.co/sd-dreambooth-library/mr-potato-head)):
```shell
git lfs install
git clone https://huggingface.co/sd-dreambooth-library/mr-potato-head
```

## Training

To fine-tune the text-to-image diffusion models for text-to-video generation, run this command:

```shell
accelerate launch train_tuneavideo.py --config="configs/man-surfing.yaml"
```

## Inference

Once the training is done, run inference:

```python
from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid
import torch

model_id = "path-to-your-trained-model"
unet = UNet3DConditionModel.from_pretrained(model_id, subfolder='unet', torch_dtype=torch.float16).to('cuda')
pipe = TuneAVideoPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", unet=unet, torch_dtype=torch.float16).to("cuda")

prompt = "a panda is surfing"
video = pipe(prompt, video_length=8, height=512, width=512, num_inference_steps=50, guidance_scale=7.5).videos

save_videos_grid(video, f"{prompt}.gif")
```

## Results

### Fine-tuning on Stable Diffusion

<table width="100%" align="center">
<tr>
  <td><img src="https://tuneavideo.github.io/static/results/man-surfing/train.gif"></td>
  <td><img src="https://tuneavideo.github.io/static/results/repo/stablediffusion/panda-surfing.gif"></td>
  <td><img src="https://tuneavideo.github.io/static/results/repo/stablediffusion/ironman-desert.gif"></td>              
  <td><img src="https://tuneavideo.github.io/static/results/repo/stablediffusion/raccoon-cartoon.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">[Training] a man is surfing.</td>
  <td width=25% style="text-align:center;">a panda is surfing.</td>
  <td width=25% style="text-align:center;">Iron Man is surfing in the desert.</td>
  <td width=25% style="text-align:center;">a raccoon is surfing, cartoon style.</td>
</tr>
</table>

### Fine-tuning on DreamBooth

<table width="100%" align="center">
<tr>
  <td><img src="https://tuneavideo.github.io/static/results/repo/dreambooth/mr-potato-head.png"></td>
  <td><img src="https://tuneavideo.github.io/static/results/repo/dreambooth/pink-hat.gif"></td>
  <td><img src="https://tuneavideo.github.io/static/results/repo/dreambooth/potato-sunglasses.gif"></td>              
  <td><img src="https://tuneavideo.github.io/static/results/repo/dreambooth/potato-forest.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">sks mr potato head.</td>
  <td width=25% style="text-align:center;">sks mr potato head, wearing a pink hat, is surfing.</td>
  <td width=25% style="text-align:center;">sks mr potato head, wearing sunglasses, is surfing.</td>
  <td width=25% style="text-align:center;">sks mr potato head is surfing in the forest.</td>
</tr>
</table>

## BibTeX
```
@article{wu2022tuneavideo,
    title={Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation},
    author={Wu, Jay Zhangjie and Ge, Yixiao and Wang, Xintao and Lei, Stan Weixian and Gu, Yuchao and Hsu, Wynne and Shan, Ying and Qie, Xiaohu and Shou, Mike Zheng},
    journal={arXiv preprint arXiv:2212.11565},
    year={2022}
}
```