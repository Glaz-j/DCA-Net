
# DCA-UNet: Dual Coordinate Attention Network


[![GitHub](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/Glaz-j/DCA-Net)

This repository provides the official PyTorch implementation for the paper:  
**"Dual Coordinate Attention (DCA) Network for Accurate Cerebral Vascular Endothelium Segmentation in OCT Images"**.

## Table of Contents
- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Key Features](#key-features)
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Usage Guide](#usage-guide)
- [Experimental Results](#experimental-results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Project Overview

This project introduces **DCA-UNet**, a novel deep learning framework specifically designed for precise segmentation of cerebrovascular endothelium in Optical Coherence Tomography (OCT) images.

Through the unique **Dual Coordinate Attention (DCA)** mechanism, our model captures features from both Cartesian and Polar coordinate systems. This design significantly enhances segmentation performance, particularly for vascular structures that are challenging for conventional methods to process due to their thin morphology and low contrast.

## Model Architecture

The core of DCA-UNet integrates our proposed DCA module into a dual-path U-Net architecture. The network takes both the original image and its polar-transformed version as dual inputs. At each encoder level, the DCA module aligns and fuses features from both paths, enabling the model to learn robust feature representations that incorporate both standard and radial patterns.

This dual-view approach effectively addresses the spatial perception limitations inherent in single-coordinate systems.

|                  DCA Module                 |             DCA-UNet Architecture           |
|:-------------------------------------------:|:-------------------------------------------:|
| <img src="doubleCoordAtt.png" width="400">  | <img src="doubleCoorAttUNet.png" width="400"> |
|       *Figure 1: Dual Coordinate Attention Module*      |       *Figure 2: DCA Integration in U-Net*       |

## Key Features

- **Dual-Domain Framework**: Processes images in both Cartesian and Polar coordinates to capture comprehensive structural information.
- **Dual Coordinate Attention (DCA)**: A lightweight and efficient attention mechanism that fuses features across coordinate systems, enhancing critical endothelial structures while suppressing noise.
- **State-of-the-art Performance**: Outperforms standard models (U-Net, U-Net++, Attention U-Net) on a challenging expert-annotated cerebrovascular OCT dataset.
- **Clinical Potential**: Provides a robust, automated analysis tool for cerebrovascular disease assessment and monitoring.

## Environment Setup

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/Glaz-j/DCA-Net.git
    cd DCA-Net
    ```

2.  **Create & Activate Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Preparation

Organize your dataset in the following directory structure:
```
data/
├── imgs/
│   ├── train_image_001.png
│   └── ...
├── masks/
│   ├── train_image_001.png
│   └── ...
├── val_imgs/
│   ├── val_image_001.png
│   └── ...
└── val_masks/
    ├── val_image_001.png
    └── ...
```
- Training scripts read data from `data/imgs` and `data/masks`
- Validation data is read from `data/val_imgs` and `data/val_masks`

## Usage Guide

### Model Training
Train `EncoderDoubleCoordAttUNet` using `train.py`:
```bash
python train.py [ARGUMENTS]
```

**Key Parameters:**
- `--epochs`: Total training epochs (default: `100`)
- `--batch-size`: Batch size (default: `1`)
- `--learning-rate`: Learning rate (default: `1e-5`)
- `--amp`: Enable Automatic Mixed Precision (reduces memory footprint)
- `--load`: Resume training from checkpoint (e.g., `checkpoints4/checkpoint_epoch90.pth`)

**Example:**
```bash
python train.py --epochs 100 --batch-size 4 --amp --learning-rate 1e-5
```
> **Note**: Checkpoints save to `./checkpoints4/` by default. Best model saved as `best_checkpoint.pth`.

---

### Model Evaluation
Evaluate trained models using `testTool.py`:

**<span style="color:red">Important</span>**: Modify the model path in `testTool.py` (line ~218) before evaluation:
```python
# Replace with your model path:
snapshot_path = "./checkpoints4/best_checkpoint.pth"
```

**Run Evaluation:**
```bash
python testTool.py
```

**Outputs:**
- **Quantitative Metrics**: Dice, HD95, Precision, etc. (printed in console and saved to `./test_log/`)
- **Visual Results**: Segmentation masks saved in `./predictions/`

## Experimental Results
Our method demonstrates state-of-the-art performance, significantly outperforming baseline models across all key metrics.

| Method         | Dice    | HD95    | Precision | Sensitivity | IoU     | VS      |
|----------------|---------|---------|-----------|-------------|---------|---------|
| U-Net          | 0.7701  | 10.4035 | 0.8196    | 0.7483      | 0.6869  | 0.8158  |
| U-Net++        | 0.8305  | 11.4765 | 0.9548    | 0.7980      | 0.7582  | 0.8497  |
| Attention UNet | 0.8044  | 12.9114 | 0.9085    | 0.7883      | 0.7375  | 0.8305  |
| **DCA-UNet (Ours)** | **0.8707** | **6.6880** | **0.8698** | **0.9053** | **0.7844** | **0.9025** |

## Citation
If using this code or ideas from our paper, please cite:
```bibtex
@article{wu2025dca,
  title={Dual Coordinate Attention (DCA) Network for Accurate Cerebral Vascular Endothelium Segmentation in OCT Images},
  author={Wu, Zhaoye and Shen, Yue and Ng, Eddie Yin Kwee and Huang, Chenxi and Lan, Quan and Ren, Lijie and Li, Jun},
  journal={Springer Nature},
  year={2025}
}
```

## Acknowledgements
This work was supported by the Open Research Fund (SCRCND202508) of the Shenzhen Clinical Research Center for Neurological Diseases (LCYSSQ20220823091204009).

