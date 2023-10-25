# Tune-A-Video

This repository is the official implementation of [Tune-A-Video](https://arxiv.org/abs/2212.11565).

**[Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation](https://arxiv.org/abs/2212.11565)**
<br/>
[Jay Zhangjie Wu](https://zhangjiewu.github.io/), 
[Yixiao Ge](https://geyixiao.com/), 
[Xintao Wang](https://xinntao.github.io/), 
[Stan Weixian Lei](), 
[Yuchao Gu](https://ycgu.site/), 
[Yufei Shi](),
[Wynne Hsu](https://www.comp.nus.edu.sg/~whsu/), 
[Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en), 
[Xiaohu Qie](https://scholar.google.com/citations?user=mk-F69UAAAAJ&hl=en), 
[Mike Zheng Shou](https://sites.google.com/view/showlab)
<br/>

[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://tuneavideo.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2212.11565-b31b1b.svg)](https://arxiv.org/abs/2212.11565)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Tune-A-Video-library/Tune-A-Video-Training-UI)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/showlab/Tune-A-Video/blob/main/notebooks/Tune-A-Video.ipynb)


<p align="center">
<img src="https://tuneavideo.github.io/assets/teaser.gif" width="1080px"/>  
<br>
<em>Given a video-text pair as input, our method, Tune-A-Video, fine-tunes a pre-trained text-to-image diffusion model for text-to-video generation.</em>
</p>

## News
### 🚨 Announcing [LOVEU-TGVE](https://sites.google.com/view/loveucvpr23/track4): A CVPR competition for AI-based video editing! Submissions due Jun 5. Don't miss out! 🤩 
- [02/22/2023] Improved consistency using DDIM inversion.
- [02/08/2023] [Colab demo](https://colab.research.google.com/github/showlab/Tune-A-Video/blob/main/notebooks/Tune-A-Video.ipynb) released!
- [02/03/2023] Pre-trained Tune-A-Video models are available on [Hugging Face Library](https://huggingface.co/Tune-A-Video-library)!
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

**[Stable Diffusion]** [Stable Diffusion](https://arxiv.org/abs/2112.10752) is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input. The pre-trained Stable Diffusion models can be downloaded from Hugging Face (e.g., [Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4), [v2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)). You can also use fine-tuned Stable Diffusion models trained on different styles (e.g, [Modern Disney](https://huggingface.co/nitrosocke/mo-di-diffusion), [Anything V4.0](https://huggingface.co/andite/anything-v4.0), [Redshift](https://huggingface.co/nitrosocke/redshift-diffusion), etc.).

**[DreamBooth]** [DreamBooth](https://dreambooth.github.io/) is a method to personalize text-to-image models like Stable Diffusion given just a few images (3~5 images) of a subject. Tuning a video on DreamBooth models allows personalized text-to-video generation of a specific subject. There are some public DreamBooth models available on [Hugging Face](https://huggingface.co/sd-dreambooth-library) (e.g., [mr-potato-head](https://huggingface.co/sd-dreambooth-library/mr-potato-head)). You can also train your own DreamBooth model following [this training example](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth). 


## Usage

### Training

To fine-tune the text-to-image diffusion models for text-to-video generation, run this command:

```bash
accelerate launch train_tuneavideo.py --config="configs/man-skiing.yaml"
```

Note: Tuning a 24-frame video usually takes `300~500` steps, about `10~15` minutes using one A100 GPU. 
Reduce `n_sample_frames` if your GPU memory is limited.

### Inference

Once the training is done, run inference:

```python
from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid
import torch

pretrained_model_path = "./checkpoints/stable-diffusion-v1-4"
my_model_path = "./outputs/man-skiing"
unet = UNet3DConditionModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')
pipe = TuneAVideoPipeline.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=torch.float16).to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()

prompt = "spider man is skiing"
ddim_inv_latent = torch.load(f"{my_model_path}/inv_latents/ddim_latent-500.pt").to(torch.float16)
video = pipe(prompt, latents=ddim_inv_latent, video_length=24, height=512, width=512, num_inference_steps=50, guidance_scale=12.5).videos

save_videos_grid(video, f"./{prompt}.gif")
```

## Results

### Pretrained T2I (Stable Diffusion)
<table class="center">
<tr>
  <td style="text-align:center;"><b>Input Video</b></td>
  <td style="text-align:center;" colspan="3"><b>Output Video</b></td>
</tr>
<tr>
  <td><img src="https://tuneavideo.github.io/assets/data/man-skiing.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/man-skiing/spiderman-beach.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/man-skiing/wonder-woman.gif"></td>              
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/man-skiing/pink-sunset.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A man is skiing"</td>
  <td width=25% style="text-align:center;">"Spider Man is skiing on the beach, cartoon style”</td>
  <td width=25% style="text-align:center;">"Wonder Woman, wearing a cowboy hat, is skiing"</td>
  <td width=25% style="text-align:center;">"A man, wearing pink clothes, is skiing at sunset"</td>
</tr>

<tr>
  <td><img src="https://tuneavideo.github.io/assets/data/rabbit-watermelon.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/rabbit-watermelon/rabbit.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/rabbit-watermelon/cat.gif"></td>              
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/rabbit-watermelon/puppy.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A rabbit is eating a watermelon on the table"</td>
  <td width=25% style="text-align:center;">"A rabbit is <del>eating a watermelon</del> on the table"</td>
  <td width=25% style="text-align:center;">"A cat with sunglasses is eating a watermelon on the beach"</td>
  <td width=25% style="text-align:center;">"A puppy is eating a cheeseburger on the table, comic style"</td>
</tr>

<tr>
  <td><img src="https://tuneavideo.github.io/assets/data/car-turn.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/car-turn/porsche-beach.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/car-turn/car-cartoon.gif"></td>              
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/car-turn/car-snow.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A jeep car is moving on the road"</td>
  <td width=25% style="text-align:center;">"A Porsche car is moving on the beach"</td>
  <td width=25% style="text-align:center;">"A car is moving on the road, cartoon style"</td>
  <td width=25% style="text-align:center;">"A car is moving on the snow"</td>
</tr>

<tr>
  <td><img src="https://tuneavideo.github.io/assets/data/man-basketball.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/man-basketball/bond.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/man-basketball/astronaut.gif"></td>              
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/man-basketball/lego.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A man is dribbling a basketball"</td>
  <td width=25% style="text-align:center;">"James Bond is dribbling a basketball on the beach"</td>
  <td width=25% style="text-align:center;">"An astronaut is dribbling a basketball, cartoon style"</td>
  <td width=25% style="text-align:center;">"A lego man in a black suit is dribbling a basketball"</td>
</tr>
</table>

### Pretrained T2I (personalized DreamBooth)

<a href="https://huggingface.co/andite/anything-v4.0"><img src="https://tuneavideo.github.io/assets/results/tuneavideo/anything-v4/anything-v4.png" width="240px"/></a>

<table class="center">
<tr>
  <td style="text-align:center;"><b>Input Video</b></td>
  <td style="text-align:center;" colspan="3"><b>Output Video</b></td>
</tr>
<tr>
  <td><img src="https://tuneavideo.github.io/assets/data/bear-guitar.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/anything-v4/bear-guitar/1girl-streets.gif"></td>      
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/anything-v4/bear-guitar/1boy-indoor.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/anything-v4/bear-guitar/1girl-beach.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A bear is playing guitar"</td>
  <td width=25% style="text-align:center;">"1girl is playing guitar, white hair, medium hair, cat ears, closed eyes, cute, scarf, jacket, outdoors, streets"</td>
  <td width=25% style="text-align:center;">"1boy is playing guitar, bishounen, casual, indoors, sitting, coffee shop, bokeh"</td>
  <td width=25% style="text-align:center;">"1girl is playing guitar, red hair, long hair, beautiful eyes, looking at viewer, cute, dress, beach, sea"</td>
</tr>
</table>


<a href="https://huggingface.co/nitrosocke/mo-di-diffusion"><img src="https://tuneavideo.github.io/assets/results/tuneavideo/modern-disney/modern-disney.png" width="240px"/></a> 

<table class="center">
<tr>
  <td style="text-align:center;"><b>Input Video</b></td>
  <td style="text-align:center;" colspan="3"><b>Output Video</b></td>
</tr>
<tr>
  <td><img src="https://tuneavideo.github.io/assets/data/bear-guitar.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/modern-disney/bear-guitar/rabbit.gif"></td>      
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/modern-disney/bear-guitar/prince.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/modern-disney/bear-guitar/princess.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A bear is playing guitar"</td>
  <td width=25% style="text-align:center;">"A rabbit is playing guitar, modern disney style"</td>
  <td width=25% style="text-align:center;">"A handsome prince is playing guitar, modern disney style"</td>
  <td width=25% style="text-align:center;">"A magic princess with sunglasses is playing guitar on the stage, modern disney style"</td>
</tr>
</table>

<a href="https://huggingface.co/sd-dreambooth-library/mr-potato-head"><img src="https://tuneavideo.github.io/assets/results/tuneavideo/mr-potato-head/mr-potato-head.png" width="240px"/></a>

<table class="center">
<tr>
  <td style="text-align:center;"><b>Input Video</b></td>
  <td style="text-align:center;" colspan="3"><b>Output Video</b></td>
</tr>
<tr>
  <td><img src="https://tuneavideo.github.io/assets/data/bear-guitar.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/mr-potato-head/bear-guitar/lego-snow.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/mr-potato-head/bear-guitar/sunglasses-beach.gif"></td>      
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/mr-potato-head/bear-guitar/van-gogh.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A bear is playing guitar"</td>
  <td width=25% style="text-align:center;">"Mr Potato Head, made of lego, is playing guitar on the snow"</td>
  <td width=25% style="text-align:center;">"Mr Potato Head, wearing sunglasses, is playing guitar on the beach"</td>
  <td width=25% style="text-align:center;">"Mr Potato Head is playing guitar in the starry night, Van Gogh style"</td>
</tr>
</table>


## Citation
If you make use of our work, please cite our paper.
```bibtex
@inproceedings{wu2023tune,
  title={Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation},
  author={Wu, Jay Zhangjie and Ge, Yixiao and Wang, Xintao and Lei, Stan Weixian and Gu, Yuchao and Shi, Yufei and Hsu, Wynne and Shan, Ying and Qie, Xiaohu and Shou, Mike Zheng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={7623--7633},
  year={2023}
}
```

## Shoutouts

- This code builds on [diffusers](https://github.com/huggingface/diffusers). Thanks for open-sourcing!
- Thanks [hysts](https://github.com/hysts) for the awesome [gradio demo](https://huggingface.co/spaces/Tune-A-Video-library/Tune-A-Video-Training-UI).
