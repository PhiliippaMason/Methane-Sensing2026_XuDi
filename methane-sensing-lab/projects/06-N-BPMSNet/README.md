# AI Methane Detection
# N-BPMSNet: NDMI-Guided Bitemporal Network for Methane Plume Detection

This repository provides the official implementation of the paper:

**D. Xu, P. J. Mason, J. Liu and Y. Wang (2026)**
*N-BPMSNet: An NDMI-Guided Bitemporal Network for Methane Plume Detection and Segmentation From Sentinel-2 Multispectral Observations*
IEEE Transactions on Geoscience and Remote Sensing (TGRS)
DOI: 10.1109/TGRS.2026.3689118

---

##  Overview

Methane plume detection from satellite imagery is critical for environmental monitoring and climate change mitigation. This project introduces **N-BPMSNet**, a novel **NDMI-guided bitemporal deep learning framework** for detecting and segmenting methane plumes from **Sentinel-2 multispectral data**.

Key features:

*  Bitemporal change detection framework
*  NDMI-guided feature enhancement
*  Self-attention-based segmentation network
*  Designed for Sentinel-2 multispectral observations

---

## Repository Structure

```
.
├── README.md
├── environment.yml
├── dataset/        # Dataset and preprocessing scripts
├── notebooks/      # Jupyter notebooks for experiments & visualization
├── scripts/        # Training and evaluation scripts
├── utils/          # Utility functions
├── network/        # Model architecture definitions
├── src/            # Core implementation
└── output/         # Results, logs, and checkpoints
```

---

## Installation

We recommend using **conda** to set up the environment:

```bash
conda env create -f environment.yml
conda activate nbpmsnet
```

---

##  Dataset

* The model is designed for **Sentinel-2 multispectral imagery**
* Preprocessing includes:

  * Radiometric correction
  * NDMI computation
  * Bitemporal pairing


---

##  Usage

```bash
sbatch scripts/slurm.sh
```

---

##  Model Architecture

N-BPMSNet consists of:

* **Bitemporal feature extractor**
* **NDMI-guided attention module**
* **Self-attention segmentation head**

Detailed implementation can be found in:

```
network/
src/
```

---

## Notebooks

Example workflows are provided in:

```
notebooks/
```

Including:

* Data preprocessing
* Compared Experiments
* Heatmap
* Model inference demo

---

##  Keywords

* Methane plume detection
* Multispectral imagery
* Sentinel-2
* Segmentation
* Self-attention
* Bitemporal learning

---

##  Citation

If you find this work useful, please cite:

```bibtex
@article{xu2026nbpmsnet,
  title={N-BPMSNet: An NDMI-Guided Bitemporal Network for Methane Plume Detection and Segmentation From Sentinel-2 Multispectral Observations},
  author={Xu, D. and Mason, P. J. and Liu, J. and Wang, Y.},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={64},
  pages={4106615--4106615},
  year={2026},
  doi={10.1109/TGRS.2026.3689118}
}
```

---

##  Contact

For questions or collaborations, please open an issue or contact the authors.

---
