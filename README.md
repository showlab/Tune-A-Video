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

[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://tuneavideo.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2212.11565-b31b1b.svg)](https://arxiv.org/abs/2212.11565)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Tune-A-Video-library/Tune-A-Video-Training-UI)
[![Hugging Face Library](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Library-green)](https://huggingface.co/spaces/Tune-A-Video-library/Tune-A-Video-Training-UI)


<p align="center">
<img src="https://tuneavideo.github.io/static/images/overview.png" width="800px"/>  
<br>
<em>Given a video-text pair, our method, Tune-A-Video, fine-tunes a pre-trained text-to-image diffusion model for text-to-video generation.</em>
</p>

## News
- [02/03/2023] Checkout our latest results tuned on [Modern Disney](https://huggingface.co/nitrosocke/mo-di-diffusion) and [Redshift](https://huggingface.co/nitrosocke/redshift-diffusion). 
- [01/28/2023] New Feature: tune a video on personalized [DreamBooth](https://dreambooth.github.io/) models.
- [01/28/2023] Code released!


## Setup

### Requirements

```shell
pip install -r requirements.txt
```

Installing [xformers](https://github.com/facebookresearch/xformers) is highly recommended for more efficiency and speed on GPUs. 
To enable xformers, set `enable_xformers_memory_efficient_attention=True` (default).

### Weights

**[Stable Diffusion]** [Stable Diffusion](https://arxiv.org/abs/2112.10752) is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input. The pre-trained Stable Diffusion models can be downloaded from Hugging Face (e.g., [Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4), [v2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)). You can also use fine-tuned Stable Diffusion models trained on different styles (e.g, [Modern Disney](https://huggingface.co/nitrosocke/mo-di-diffusion), [Redshift](https://huggingface.co/nitrosocke/redshift-diffusion), etc.).

**[DreamBooth]** [DreamBooth](https://dreambooth.github.io/) is a method to personalize text-to-image models like Stable Diffusion given just a few images (3~5 images) of a subject. Tuning a video on DreamBooth models allows personalized text-to-video generation of a specific subject. There are some public DreamBooth models available on [Hugging Face](https://huggingface.co/sd-dreambooth-library) (e.g., [mr-potato-head](https://huggingface.co/sd-dreambooth-library/mr-potato-head)). You can also train your own DreamBooth model following [this training example](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth). 


## Usage

### Training

To fine-tune the text-to-image diffusion models for text-to-video generation, run this command:

```bash
accelerate launch train_tuneavideo.py --config="configs/man-surfing.yaml"
```

Note: Tuning a video usually takes `300~500` steps, about `5~10` minutes using one A100 GPU and `10~20` minutes using one V100 GPU.

### Inference

Once the training is done, run inference:

```python
from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid
import torch

pretrained_model_path = "./checkpoints/stable-diffusion-v1-4"
unet_model_path = "./outputs/man-surfing/2023-XX-XXTXX-XX-XX"
unet = UNet3DConditionModel.from_pretrained(unet_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')
pipe = TuneAVideoPipeline.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=torch.float16).to("cuda")
pipe.enable_xformers_memory_efficient_attention()

prompt = "a panda is surfing"
video = pipe(prompt, video_length=8, height=512, width=512, num_inference_steps=50, guidance_scale=7.5).videos

save_videos_grid(video, f"./{prompt}.gif")
```

## Results

### [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4)

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

### [Mr Potato Head](https://huggingface.co/sd-dreambooth-library/mr-potato-head)

<table width="100%" align="center">
<tr>
  <td><img src="https://tuneavideo.github.io/static/results/repo/dreambooth/mr-potato-head.png"></td>
  <td><img src="https://tuneavideo.github.io/static/results/repo/dreambooth/pink-hat.gif"></td>
  <td><img src="https://tuneavideo.github.io/static/results/repo/dreambooth/potato-sunglasses.gif"></td>              
  <td><img src="https://tuneavideo.github.io/static/results/repo/dreambooth/potato-forest.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">[DreamBooth] sks mr potato head.</td>
  <td width=25% style="text-align:center;">sks mr potato head, wearing a pink hat, is surfing.</td>
  <td width=25% style="text-align:center;">sks mr potato head, wearing sunglasses, is surfing.</td>
  <td width=25% style="text-align:center;">sks mr potato head is surfing in the forest.</td>
</tr>
</table>


### [Modern Disney](https://huggingface.co/nitrosocke/mo-di-diffusion)

<table width="100%" align="center">
<tr>
  <td><img src="https://tuneavideo.github.io/static/results/bear-guitar/train.gif"></td>
  <td><img src="https://tuneavideo.github.io/static/results/repo/modern-disney/prince-guitar.gif"></td>
  <td><img src="https://tuneavideo.github.io/static/results/repo/modern-disney/princess-guitar.gif"></td>              
  <td><img src="https://tuneavideo.github.io/static/results/repo/modern-disney/rabbit-guitar.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">[Training] a bear is playing guitar.</td>
  <td width=25% style="text-align:center;">a handsome prince is playing guitar, modern disney style.</td>
  <td width=25% style="text-align:center;">a magical princess is playing guitar on the beach, modern disney style.</td>
  <td width=25% style="text-align:center;">a rabbit is playing guitar, modern disney style.</td>
</tr>
</table>

### [Redshift](https://huggingface.co/nitrosocke/redshift-diffusion)

<table width="100%" align="center">
<tr>
  <td><img src="https://tuneavideo.github.io/static/results/man-skiing/train.gif"></td>
  <td><img src="https://tuneavideo.github.io/static/results/repo/redshift/spiderman-skiing.gif"></td>
  <td><img src="https://tuneavideo.github.io/static/results/repo/redshift/batman-skiing.gif"></td>              
  <td><img src="https://tuneavideo.github.io/static/results/repo/redshift/hulk-skiing.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">[Training] a man is skiing.</td>
  <td width=25% style="text-align:center;">spider man is skiing.</td>
  <td width=25% style="text-align:center;">bat man is skiing.</td>
  <td width=25% style="text-align:center;">hulk is skiing.</td>
</tr>
</table>


## Citation
If you make use of our work, please cite our paper.
```bibtex
@article{wu2022tuneavideo,
    title={Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation},
    author={Wu, Jay Zhangjie and Ge, Yixiao and Wang, Xintao and Lei, Stan Weixian and Gu, Yuchao and Hsu, Wynne and Shan, Ying and Qie, Xiaohu and Shou, Mike Zheng},
    journal={arXiv preprint arXiv:2212.11565},
    year={2022}
}
```

## Shoutouts

- This code builds on [diffusers](https://github.com/huggingface/diffusers). Thanks for open-sourcing!
- Thanks [hysts](https://github.com/hysts) for the awesome [gradio demo](https://huggingface.co/spaces/Tune-A-Video-library/Tune-A-Video-Training-UI).