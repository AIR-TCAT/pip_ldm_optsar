
# PIP-LDM: Physics-Informed Parametric Latent Diffusion for Optical-SAR Cross-Modal Image Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

[cite_start]This repository contains the official PyTorch implementation of **PIP-LDM**[cite: 10]. [cite_start]It is a physics-informed parametric latent diffusion model designed for high-fidelity cross-modal generation between optical and Synthetic Aperture Radar (SAR) images[cite: 10, 115]. 

[cite_start]Unlike conventional data-driven generative models that often reduce cross-modal mapping to statistical style transfer, PIP-LDM explicitly bridges the gap between passive reflectance (optical) and active backscatter (SAR) imaging mechanisms[cite: 8, 23]. 

## 🌟 Key Innovations

[cite_start]To address the common issues of mislocalized scattering centers, distorted speckle statistics, and structural degradation in latent spaces, PIP-LDM introduces three novel modules[cite: 9, 52]:

* [cite_start]**Physical Feature Injection (PFI):** A dual-stream architecture (CLIP + BERT) that systematically incorporates imaging-condition descriptors (e.g., satellite platform, spatial resolution) and scene-level object semantics into the generative process[cite: 11, 143, 149, 152].
* **Multi-level Physical Consistency Constraint (MPC):** A regularization module that enforces physical realism across three scales:
    * [cite_start]*Macro-scale:* Geometric structure alignment using gradient-domain constraints[cite: 12, 173].
    * [cite_start]*Feature-scale:* Dominant scattering center preservation via a statistically thresholded mask[cite: 12, 180].
    * [cite_start]*Micro-scale:* Speckle statistics consistency by matching latent feature moments (mean and variance)[cite: 12, 195].
* [cite_start]**Latent Regularization Loss Module (LRLM):** Mitigates the low-pass filtering effect of standard VAEs by applying an $L_1$-norm sparsity penalty and Gram matrix-based connectivity regularization to preserve SAR-specific sparse structures in the latent space[cite: 13, 215, 222].

## 📊 Supported Datasets

[cite_start]PIP-LDM has been extensively evaluated and generalizes well across diverse scenes and spatial resolutions (from 1m to 10m)[cite: 14]. Supported datasets include:

* [cite_start]**SEN12:** 10m resolution, Sentinel-1 / Sentinel-2 pairs[cite: 248, 252].
* [cite_start]**WHU-OPT-SAR:** 5m resolution, Gaofen-3 / Gaofen-1 pairs[cite: 249, 252].
* [cite_start]**QXS-SAROPT:** 1m resolution, Gaofen-3 / Google Earth pairs[cite: 248, 252].
* [cite_start]**OSDataset:** 1m resolution, Sentinel-1 / Google Earth pairs[cite: 249, 252].

## 🚀 Performance Highlights

[cite_start]Compared to previous state-of-the-art methods (both GAN-based and Diffusion-based), PIP-LDM demonstrates superior physical consistency and visual fidelity[cite: 62]:
* [cite_start]Reduces Fréchet Inception Distance (FID) by an average of **4.63%** for Optical-to-SAR (O2S) tasks[cite: 15].
* [cite_start]Reduces FID by an average of **13.13%** for SAR-to-Optical (S2O) tasks[cite: 15].
* [cite_start]Significantly improves Semantic-Supervised Rectified IoU (SSR-IoU) for precise scattering-center localization and Kullback-Leibler Divergence (KLD) for authentic speckle statistics[cite: 16, 255].

## 🛠️ Installation

**1. Clone the repository:**
```bash
git clone https://github.com/YourUsername/PIP-LDM.git
cd PIP-LDM
```

**2. Create a virtual environment and install dependencies:**
```bash
conda create -n pip-ldm python=3.9
conda activate pip-ldm
pip install -r requirements.txt
```

## 📂 Data Preparation

Please organize your datasets in the following structure before training:

```text
dataset_name/
├── train/
│   ├── opt_0001.png
│   ├── sar_0001.png
├── val/
├── test/
└── train_captions.json  # Containing scene semantics and imaging conditions
```

## 💻 Usage

### Training
To train the PIP-LDM model from scratch on your dataset (e.g., O2S translation), run the following command:

```bash
accelerate launch train.py \
  --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
  --train_json_path "./dataset/train_captions.json" \
  --val_json_path "./dataset/val_captions.json" \
  --output_dir "./outputs" \
  --train_batch_size 16 \
  --learning_rate 1e-5
```

## 📖 Citation

[cite_start]If you find this code or our paper useful for your research, please cite our work[cite: 1, 2, 3, 4, 5, 6]:

```bibtex

```

## 📄 License
This project is released under the [MIT License](LICENSE).