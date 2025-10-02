# TSGNet

This repository contains the official implementation of the paper: "<b>Angular Sparsity Invariant Tilt Series Generation in Scanning/Transmission Electron Microscopy</b>", accepted at the <b>NeurIPS 2025 Machine Learning for Physical Sciences (ML4PS) Workshop</b>.


## Overview
Electron tomography (ET) reconstructs 3D nanostructures from a series of 2D Scanning/Transmission Electron Microscopy (S/TEM) tilt images. However, radiation-sensitive materials restrict the number of tilt images that can be acquired, leading to sparse angular sampling, artifacts, and reduced resolution.

We introduce TSGNet, a deep convolutional encoder–decoder framework designed to predict missing tilt images in sparse tilt series. Our method recovers intermediate projections, enabling improved 3D reconstructions while reducing acquisition dose and time.

Key contributions:

- Large-scale simulated S/TEM dataset with aberrated and non-aberrated tilt series.

- TSGNet (Tilt Series Generation Network), a pyramid encoder–decoder for tilt frame interpolation.

- Significant improvements in MSE, PSNR, and SSIM over baselines.

- Demonstrated enhanced 3D reconstructions using predicted frames.



## Repository Structure

```bash
├──datamodules/                # Pytorch Lightning Datamodule files for the Dataloader.
├──datasets/                   # Pytorch Dataset definitiions.
├──models/                     # TSGNet model architecture file.
├──scripts/                    # Training scripts
├──trainers/                   # Pytorch Lightning TSGNet Trainer file.
├──losses.py                   # Losses defined in the paper.
├──metrics.py                  # Metric definitions for PSNR and SSIM.
├──README.md                   # This file.
└──utils.py                    # Some utility function definitions.
```

## Installation
Clone the repository and install dependencies. We recommend using [conda](https://docs.conda.io/) for environment setup.  

```bash
git clone https://github.com/dikshantsagar/TSGNet.git
cd TSGNet
conda create -n tsgnet python=3.10
conda activate tsgnet
pip install -r requirements.txt
```

## Dataset

Coming Soon...


## Training

To train TSGNet on the aberrated dataset:
```bash
torchrun --nproc_per_node=4 scripts/train_ab.py --learning_rate 1e-4 --weight_decay 0.1 --batch_size 24 --gpus 4 --max_epochs 100 --downsample_rate {temporal_sparsity_rate} --name {experiment_name}
```

To train TSGNet on the nn-aberrated dataset:
```bash
torchrun --nproc_per_node=4 scripts/train_non-ab.py --learning_rate 1e-4 --weight_decay 0.1 --batch_size 24 --gpus 4 --max_epochs 100 --downsample_rate {temporal_sparsity_rate} --name {experiment_name}
```


## Results
| Method               | Aberrated MSE ↓ | Aberrated PSNR ↑ | Aberrated SSIM ↑ | Non-aberrated MSE ↓ | Non-aberrated PSNR ↑ | Non-aberrated SSIM ↑ |
| -------------------- | --------------- | ---------------- | ---------------- | ------------------- | -------------------- | -------------------- |
| Linear Interpolation | 7.48e-9         | 30.48            | 0.90             | 1.13e-8             | 28.57                | 0.86                 |
| **TSGNet (Ours)**    | **5.77e-9**     | **31.63**        | **0.93**         | **7.13e-9**         | **29.43**            | **0.89**             |


## Citation
If you use this code or dataset, please cite our paper:
```graphql
TODO:
```
