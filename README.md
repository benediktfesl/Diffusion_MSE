# On the Asymptotic Mean Square Error Optimality of Diffusion Models

Implementation to reproduce the simulation results of the paper
>B. Fesl, B. Böck, F. Strasser, M. Baur, M. Joham, W. Utschick, "On the Asymptotic Mean Square Error Optimality of Diffusion Models." In _Proceedings of the 28th International Conference on Artificial Intelligence and Statistics_, 2025.

[[arXiv](https://arxiv.org/abs/2403.02957)] [[OpenReview](https://openreview.net/forum?id=XrXlAYFpvR)] [[PMLR](https://proceedings.mlr.press/v258/fesl25a.html)]

---

## DMSE Scheduler Package

A standalone, reusable implementation of the DMSE scheduler is available as a Python package:

```bash
pip install diffusers-dmse
```

- PyPI: https://pypi.org/project/diffusers-dmse/
- Source: https://github.com/benediktfesl/diffusers-MSEopt

This repository contains the original research implementation used for the experiments in the paper.  
The PyPI package provides a clean, diffusers-compatible version of the scheduler, but is **not used directly in this codebase**.

---

## Load data
Load data and pre-trained models from 
https://syncandshare.lrz.de/getlink/fiAsDStAV6i5FFJHyfhrcY/ (Passcode: Diffusion2024)
and move it to the project's directory.

## Information about required packages
Down below is a list of mandatory and optional packages with their versions, if a specific version is required.

### Mandatory packages
* conda-build
* cudatoolkit=11.8
* numpy
* pytorch=2.0.1
* torchvision=0.15.2=py310_cu118 (only if working with image data)
* pytorch-cuda=11.8
* scikit-learn
* scipy
* tqdm
* pip
* pytorch-fid==0.3
* matplotlib

### Optional packages (for jupyter notebooks or special script options)
* seaborn
* ipykernel
* ray-core
* ray-tune
* ray-train
* ray-dashboard

## Common Usage 
1. Train and test a DPM-based denoiser on GPU
```shell
python dpm_denoiser.py -d cuda:0
```

2. Load pre-trained model and evaluate it
```shell
python load_and_eval_dpm.py -d cuda:0
```

3. Evaluate real-valued baselines (GMM-based CME and LS)
```shell
python baselines.py
```

4. Evaluate complex-valued baseline with audio-data (GMM-based CME and LS)
```shell
python audio_gmm.py
```

## Data Options
1. rand_gmm
2. MNIST_gmm
3. FASHION_MNIST_gmm
4. audio_gmm

## Licenses

The diffusion model architecture is based upon the code from https://github.com/hojonathanho/diffusion.

The real-valued Gaussian mixture model implementation stems from https://scikit-learn.org/stable/modules/mixture.html and is covered by the BSD 3-Clause License.

The complex-valued extension of the Gaussian mixture model implementation stems from https://github.com/benediktfesl/GMM_cplx and is covered by the BSD 3-Clause License.
