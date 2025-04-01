# RefEdit: A Benchmark and Method for Improving Instruction-based Image Editing Model for Referring Expression

[Bimsara Pathiraja](https://scholar.google.es/citations?hl=en&user=7ViSGnIAAAAJ), [Malitha Gunawardhana](https://scholar.google.com/citations?user=z--mlKgAAAAJ&hl=en&oi=ao), [Shivam Singh](https://scholar.google.com/citations?user=z--mlKgAAAAJ&hl=en&oi=ao), [Yezhou Yang](https://scholar.google.com/citations?user=k2suuZgAAAAJ&hl=en), [Chitta Baral](https://scholar.google.com/citations?user=9Yd716IAAAAJ&hl=en)

[![paper](https://img.shields.io/badge/arXiv-Paper-42FF33)]() 
[![Project Page](https://img.shields.io/badge/Project-Page-blue)]() 

> **Abstract:** *Despite recent advances in inversion and instruction-based image editing, existing approaches primarily excel at editing single, prominent objects but significantly struggle when applied to complex scenes containing multiple entities. To quantify this gap, we first introduce RefEdit-Bench, a rigorous real-world benchmark rooted in RefCOCO, where even baselines trained on millions of samples perform poorly. To overcome this limitation, we introduce RefEdit -- an instruction-based editing model trained on our scalable synthetic data generation pipeline. Our RefEdit, trained on only 20,000 editing triplets, outperforms the Flux/SD3 model-based baselines trained on millions of data. Extensive evaluations across various benchmarks demonstrate that our model not only excels in referring expression tasks but also enhances performance on traditional benchmarks, achieving state-of-the-art results comparable to closed-source methods. We will release our code, data, and checkpoints.*

<p align="center">
    <figure>
        <img src="imgs/eval_page-0001.jpg" alt="Evaluation Results">
        <figcaption><b>Figure 1:</b> RefEdit is a referring expression-based image editing benchmark and a finetuned model. Our proposed RefEdit model can
accurately identify the entity of interest and perform accurate edits.</figcaption>
    </figure>
</p>

# Introduction

RefEdit is a benchmark and method for improving instruction-based image editing models for referring expressions. The goal of RefEdit is to enhance the performance of image editing models when dealing with complex scenes containing multiple entities. The benchmark is built upon the RefCOCO dataset, which contains referring expressions for various objects in images.

We also provide the finetuned models on InstructPix2Pix and UltraEdit-freeform - RefEdit-SD1.5 and RefEdit-SD3, respectively. The models are trained on our synthetic data generation pipeline, which generates high-quality editing triplets for training combined with MagicBrush data. The models are evaluated on both our RefEdit-Bench and PIE-Bench benchmarks using VIEScore and human evaluation. The results show that our models outperform existing baselines and achieve state-of-the-art performance on various benchmarks.

# Setup

To download the necessary GitHub repositories and copy the new files to the correct locations, run the following command:

```
bash setup.sh    
```

# Dataset Access

The training set and the RefEdit-Bench are publicly available on [Huggingface](https://huggingface.co/). The training dataset covers a synthetically generated ~20K editing triplets on changing color, changing object, adding an object, removing an object and changing texture. [MagicBrush](https://github.com/OSU-NLP-Group/MagicBrush) was combined with our synthetic dataset for final training. RefEdit-Bench is a benchmark for evaluating the performance of image editing models on referring expressions. It consists of Easy and Hard mode where each category contains 100 images. 

# Model Access

Fine-tuned checkpoints RefEdit-SD1.5 (finetuned on [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix/tree/main)) and RefEdit-SD3 (finetuned on [UltraEdit-freeform](https://github.com/pkunlp-icler/UltraEdit/tree/main?tab=readme-ov-file)) are available on [Huggingface](https://huggingface.co/). 

## 1. Environment Setup
Follow the instructions at `training/RefEdit-SD3/UltraEdit` to set up the environment for training and inference.

## 2. Inference
To run inference, use the following command:

```python
# For Editing with RefEdit-SD3
import torch
from diffusers import StableDiffusion3InstructPix2PixPipeline
from diffusers.utils import load_image
import requests
import PIL.Image
import PIL.ImageOps

pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained("RefEdit/RefEdit-SD3", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
prompt = "Add a flower bunch to the person with a red jacket"
img = load_image("RefEdit/imgs/person_with_red_jacket.jpg").resize((512, 512))

image = pipe(
    prompt,
    image=img,
    mask_img=None,
    num_inference_steps=50,
    image_guidance_scale=1.5,
    guidance_scale=7.5,
).images[0]

image.save("RefEdit/imgs/edited_image.png")
```

## 3. Training

To train the RefEdit-SD3 model, use the following command:

```
cd training/RefEdit-SD3/UltraEdit
bash scripts/run_sft_512_sd3_stage1_refedit_wo_mask.sh
```

# Citation

If you use this code or the dataset in your research, please cite our paper:

```bibtex
@article{pathiraja2023refedit,
  title={RefEdit: A Benchmark and Method for Improving Instruction-based Image Editing Model for Referring Expression},
  author={Pathiraja, Bimsara and Patel, Maitreya and Singh, Shivam and Yang, Yezhou and Baral, Chitta},
  journal={arXiv preprint arXiv:2309.12345},
  year={2025}
}
```