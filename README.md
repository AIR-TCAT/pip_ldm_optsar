PIP-LDM: Physics-Informed Parametric Latent Diffusion for Optical-SAR Cross-Modal Image GenerationThis repository contains the official PyTorch implementation of PIP-LDM. It is a physics-informed parametric latent diffusion model designed for high-fidelity cross-modal generation between optical and Synthetic Aperture Radar (SAR) images.Unlike conventional data-driven generative models that often reduce cross-modal mapping to statistical style transfer, PIP-LDM explicitly bridges the gap between passive reflectance (optical) and active backscatter (SAR) imaging mechanisms.🌟 Key InnovationsTo address the common issues of mislocalized scattering centers, distorted speckle statistics, and structural degradation in latent spaces, PIP-LDM introduces three novel modules:Physical Feature Injection (PFI): A dual-stream architecture (CLIP + BERT) that systematically incorporates imaging-condition descriptors (e.g., satellite platform, spatial resolution) and scene-level object semantics into the generative process.Multi-level Physical Consistency Constraint (MPC): A regularization module that enforces physical realism across three scales:Macro-scale: Geometric structure alignment using gradient-domain constraints.Feature-scale: Dominant scattering center preservation via a statistically thresholded mask.Micro-scale: Speckle statistics consistency by matching latent feature moments (mean and variance).Latent Regularization Loss Module (LRLM): Mitigates the low-pass filtering effect of standard VAEs by applying an $L_1$-norm sparsity penalty and Gram matrix-based connectivity regularization to preserve SAR-specific sparse structures in the latent space.📊 Supported DatasetsPIP-LDM has been extensively evaluated and generalizes well across diverse scenes and spatial resolutions (from 1m to 10m). Supported datasets include:SEN12: 10m resolution, Sentinel-1 / Sentinel-2 pairs.WHU-OPT-SAR: 5m resolution, Gaofen-3 / Gaofen-1 pairs.QXS-SAROPT: 1m resolution, Gaofen-3 / Google Earth pairs.OSDataset: 1m resolution, Sentinel-1 / Google Earth pairs.🚀 Performance HighlightsCompared to previous state-of-the-art methods (both GAN-based and Diffusion-based), PIP-LDM demonstrates superior physical consistency and visual fidelity:Reduces Fréchet Inception Distance (FID) by an average of 4.63% for Optical-to-SAR (O2S) tasks.Reduces FID by an average of 13.13% for SAR-to-Optical (S2O) tasks.Significantly improves Semantic-Supervised Rectified IoU (SSR-IoU) for precise scattering-center localization and Kullback-Leibler Divergence (KLD) for authentic speckle statistics.🛠️ Installation1. Clone the repository:Bashgit clone https://github.com/TCAT-tfj/pip_ldm_optsar.git
cd PIP-LDM
2. Create a virtual environment and install dependencies:Bashconda create -n pip-ldm python=3.9
conda activate pip-ldm
pip install -r requirements.txt
📂 Data PreparationPlease organize your datasets in the following structure before training:Plaintextdataset_name/
├── train/
│   ├── opt_0001.png
│   ├── sar_0001.png
├── val/
├── test/
└── train_captions.json  # Containing scene semantics and imaging conditions
💻 UsageTrainingTo train the PIP-LDM model from scratch on your dataset (e.g., O2S translation), run the following command:Bashaccelerate launch train.py \
  --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
  --train_json_path "./dataset/train_captions.json" \
  --val_json_path "./dataset/val_captions.json" \
  --output_dir "./outputs" \
  --train_batch_size 16 \
  --learning_rate 1e-5
InferenceTo generate SAR images from optical inputs using a trained checkpoint:Bashpython inference.py \
  --model_path "./outputs/checkpoint-final" \
  --input_image "./dataset/test/opt_sample.png" \
  --prompt "A 1m resolution Gaofen-3 SAR image of a dense urban block" \
  --output_path "./results/sar_generated.png"
📖 CitationIf you find this code or our paper useful for your research, please cite our work:Code snippet@article{hou2026pipldm,
  title={Physics-Informed Parametric Latent Diffusion for Optical-SAR Cross-Modal Image Generation},
  author={Hou, Yingyan and Tan, Fangjie and Ren, Chao and Lu, Wanxuan and Huang, Yuhong and Yu, Hongfeng and Li, Ya and Wang, Yixiao and Sun, Xian},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026}
}
📄 LicenseThis project is released under the MIT License.