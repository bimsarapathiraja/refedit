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

```
pip install -r requirements

cd diffusers && pip install -e .
```