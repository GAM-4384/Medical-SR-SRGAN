# Medical-SR-SRGAN

**Improved SRGAN for Signer-Independent Medical X-Ray Image Super-Resolution Reconstruction**

*Python Program Design Course Project — Shanghai Maritime University, 2024*

---

## Overview

This project investigates super-resolution reconstruction of low-dose chest X-ray images using an improved SRGAN framework. Starting from a baseline XRaySR network, five model variants are developed and compared through progressive architectural and training improvements. The final model achieves PSNR 28.58 dB and SSIM 0.8476 on a subset of the NIH Chest X-ray dataset while reducing parameter count by 33% relative to the standard residual block baseline.

The system includes a full data preprocessing pipeline, five independently trainable model variants, multi-model comparison tooling, and a PyQt5 desktop application for interactive inference.

---

## Model Evolution

The five variants under `Tes/` represent a progressive research trajectory, each building on lessons from the previous iteration.

```
tes0_Root               XRaySR (Baseline)
    Standard residual blocks × 4
    Charbonnier + SSIM loss
    4× PixelShuffle upsampling
    Adaptive enhancement mask
        │
        ▼
tes1_Add_Mse            XRaySR + MSE Loss
    CombinedLoss: Charbonnier(w=1.0) + SSIM(w=0.1) + MSE(w=0.1)
    Direct PSNR optimisation via MSE term
    Finding: MSE suppresses fine structural detail
        │
        ▼
tes2_EnhancedResidualBlock   OptimizedXRaySR
    Depthwise separable convolutions
    SE (Squeeze-and-Excitation) channel attention
    VGG perceptual loss + gradient checkpointing
    Progressive 2× upsampling strategy
        │
        ▼
tes3_Refer_to_Vgg       Full VGG Perceptual Loss (Abandoned)
    Full VGG-16 feature extractor in loss
    Finding: GPU OOM on target hardware; impractical
        │
        ▼
tes4_Light_Vgg          Lightweight BalancedXRaySR (Final)
    Hourglass residual block: channels → channels/2 → channels
    Grouped convolution (groups = channels/2)
    Lightweight channel attention
    Residual blocks: 4 → 6  |  Params/block: 18,432 → ~1,808
    Step-wise 2×+2× upsampling with intermediate cache flush
    BalancedLoss: L1 + SSIM (no MSE)
    PSNR 28.58 dB  |  SSIM 0.8476
```

---

## Architecture (Final Model)

```
Input LR X-ray [B, 1, H, W]
        │
   First Conv 7×7 + PReLU
        │
   6 × LightweightResidualBlock
   ├─ 1×1 Conv (channels → channels/2)
   ├─ BN + PReLU
   ├─ 3×3 Group Conv (groups = channels/2)
   ├─ BN + PReLU
   ├─ 1×1 Conv (channels/2 → channels)
   ├─ BN
   ├─ Channel Attention
   └─ Learnable scale factor on residual
        │
   Feature Fusion Conv + BN
        │
   Global residual connection
        │
   2× PixelShuffle + PReLU
   → torch.cuda.empty_cache()
   2× PixelShuffle + PReLU
        │
   Final Conv → SR output [B, 1, 4H, 4W]
        │
   Adaptive Enhancement Mask (Sigmoid gate)
        │
Output SR X-ray [B, 1, 4H, 4W]
```

**Loss:** `BalancedLoss = λ_L1 · L1 + λ_SSIM · (1 − SSIM)`

**Training:** OneCycleLR (warmup 30% → peak → cooldown), AdamW, mixed precision (AMP), gradient clipping (norm=0.5), ModelEMA with dynamic decay.

---

## Dataset

Experiments use approximately 28,000 grayscale chest X-ray images (1024×1024 px) from the NIH Chest X-ray dataset, covering 14 pathology categories. The dataset is not included in this repository.

Preprocessing (`preprocess.py`) simulates low-dose acquisition by applying, in order: contrast reduction → Gaussian noise → Gaussian blur → 4× bicubic downsampling. The resulting HR/LR image pairs are organised as:

```
processed/
├── train/
│   ├── hr/     # High-resolution ground truth
│   └── lr/     # Simulated low-dose input
├── val/
│   ├── hr/
│   └── lr/
└── test/
    ├── hr/
    └── lr/
```

---

## Project Structure

```
PY-SRRESNET/
│
├── Tes/                            # Five model variants (research trajectory)
│   ├── tes0_Root/                  # Baseline XRaySR
│   │   ├── config.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   └── train.py
│   ├── tes1_Add_Mse/               # + MSE loss term
│   │   └── ...
│   ├── tes2_EnhancedResidualBlock/ # Depthwise separable + SE attention
│   │   └── ...
│   ├── tes3_Refer_to_Vgg/          # Full VGG perceptual loss (OOM, abandoned)
│   │   └── ...
│   └── tes4_Light_Vgg/             # Lightweight hourglass residual (final)
│       └── ...
│
├── tes2/                           # Auxiliary experiment outputs
├── tes3 vgg/                       # Auxiliary experiment outputs
│
├── app.py                          # PyQt5 desktop inference application
├── compare_models.py               # Multi-model comparison and visualisation
├── model_test.py                   # Evaluation framework (PSNR/SSIM/UIQI/etc.)
├── preprocess.py                   # Data preprocessing and degradation pipeline
├── training.log                    # Training log
├── requirement.txt                 # Python dependencies
│
├── checkpoints/                    # Saved model weights
├── logs/                           # TensorBoard logs
├── outputs/                        # Inference outputs
├── processed/                      # Preprocessed HR/LR image pairs
├── raw/                            # Raw NIH dataset (not tracked)
└── runs/                           # TensorBoard run directories
```

---

## Training

Each model variant is self-contained. To train the final model:

```bash
cd Tes/tes4_Light_Vgg
python train.py
```

To train from a specific variant, navigate to the corresponding subdirectory and run `train.py`. All hyperparameters are managed through each variant's `config.py`.

---

## Evaluation

`model_test.py` evaluates any trained checkpoint against the test set and reports six metrics: PSNR, SSIM, L1, NRMSE, Perceptual Loss, and UIQI. Results are saved as a structured JSON file alongside violin plots and image triplet comparisons (LR / SR / HR).

```bash
python model_test.py
```

## Multi-Model Comparison

`compare_models.py` loads multiple checkpoints and produces side-by-side violin plots, box plots, and a statistical summary (independent t-tests on PSNR, SSIM, UIQI) across all selected models.

```bash
python compare_models.py
```

## Desktop Application

`app.py` launches a PyQt5 GUI with three panels (input / controls / output). Load a model checkpoint from the sidebar, upload any PNG/JPG/BMP X-ray image, and export the 4× super-resolved result.

```bash
python app.py
```

---

## Key Results

| Model | PSNR (dB) | SSIM | Notes |
|---|---|---|---|
| tes0_Root (Baseline) | — | — | Reference |
| tes1_Add_Mse | — | — | MSE hurts structure |
| tes2_EnhancedResidualBlock | — | — | VGG loss, higher cost |
| tes3_Refer_to_Vgg | — | — | OOM, abandoned |
| **tes4_Light_Vgg (Final)** | **28.58** | **0.8476** | 33% fewer params |

Processing throughput on RTX 3060: **8.15 images/sec** (AMP enabled).

---

## License

This repository is released for academic and non-commercial use only.

Copyright © 2024 顾淅元, 李晓宇 — Shanghai Maritime University. All rights reserved.

Redistribution or reuse in other course submissions or academic competitions is not permitted. Non-commercial academic reference with attribution is welcome.

THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND.
