# Methane Sensing Lab

This repository provides a clean workflow for methane plume detection using hyperspectral EMIT observations, PRISMA hyperspectral observations, and multispectral Sentinel-2 data. The main processing pipeline is based on matched-filter methane enhancement retrieval.

The repository supports reproducible methane plume detection experiments, including EMIT radiance preprocessing, PRISMA hyperspectral preprocessing, methane target spectrum generation, MAG1C matched-filter retrieval, Sentinel-2 methane enhancement mapping, and contrast-based visual analysis.

## Overview

Methane point-source emissions can be detected from remote sensing observations by enhancing weak methane absorption features against the background surface reflectance or radiance. This repository focuses on three data sources:

* **EMIT hyperspectral data**, preprocessed using `emit-utils` and processed using MAG1C.
* **PRISMA hyperspectral data**, converted from HE5 to ENVI format and processed using MAG1C with a sensor-specific methane target spectrum.
* **Sentinel-2 multispectral data**, processed using a customized `mag1c_s2.py` workflow.

The methane retrieval step is based on the public MAG1C implementation developed by Foote et al. The original MAG1C repository is available at:

https://github.com/markusfoote/mag1c

The EMIT preprocessing step uses tools from the EMIT utils science data system repository. As described by the EMIT SDS project:

> Welcome to the EMIT utils science data system repository. To understand how this repository is linked to the rest of the emit-sds repositories, please see the repository guide.

Repository guide:

https://github.com/emit-sds/emit-main/wiki/Repository-Guide

## Repository Structure

```text
methane-sensing-lab/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── scripts/
│   ├── run_emit_case1.sh
│   ├── run_s2_case1.sh
│   └── mag1c_s2.py
│
├── prisma/
│   ├── convert_prisma.R
│   ├── prisma.ipynb
│   └── prisma.sh
│
├── notebooks/
│   └── Contrast_exp.ipynb
│
├── data/
│   └── README.md
│
├── outputs/
│   └── README.md
│
└── docs/
    ├── workflow.md
    └── prisma_workflow.md
```

Large raw data and generated products are not included in this repository. Please place input data under a local data directory and save processing results under an output directory.

## Installation

Clone this repository:

```bash
git clone https://github.com/<your-username>/methane-sensing-lab.git
cd methane-sensing-lab
```

MAG1C and EMIT utility tools should be installed or cloned separately. For example:

```bash
cd /mnt/d/Project4/code
git clone https://github.com/markusfoote/mag1c.git
```

The EMIT preprocessing utility should also be available locally as `emit-utils`. Please refer to the EMIT SDS repository guide for how the EMIT utility repository is connected with the rest of the EMIT SDS repositories:

https://github.com/emit-sds/emit-main/wiki/Repository-Guide

## EMIT Data Workflow

The EMIT workflow contains three main steps:

1. Reformat EMIT observation and radiance files.
2. Run MAG1C methane matched-filter retrieval.
3. Visualize and compare the results using the contrast experiment notebook.

### Step 1: Reformat EMIT Files

Go to the `emit-utils` directory:

```bash
cd /mnt/d/Project4/code/emit-utils
```

Reformat the EMIT observation file:

```bash
python -m emit_utils.reformat \
  ../../EMIT_data/EMIT_L1B_OBS_001_20240409T074435_2410005_007.nc \
  ./output/case1/
```

Reformat the EMIT radiance file:

```bash
python -m emit_utils.reformat \
  ../../EMIT_data/EMIT_L1B_RAD_001_20240409T074435_2410005_007.nc \
  ./output/case1/
```

### Step 2: Run MAG1C on EMIT Radiance Data

Go to the MAG1C directory:

```bash
cd /mnt/d/Project4/code/mag1c/mag1c
```

Run MAG1C:

```bash
python mag1c.py \
  ../../emit-utils/output/case1/EMIT_L1B_RAD_001_20240409T074435_2410005_007_radiance \
  --out ../../res/case1/20240409T074435_emit_mf \
  -o
```

The output will be saved under:

```text
/mnt/d/Project4/code/res/case1/
```

### Step 3: Visualization and Contrast Experiment

Open the notebook:

```text
notebooks/Contrast_exp.ipynb
```

This notebook can be used to compare methane enhancement maps, Sentinel-2 results, EMIT matched-filter outputs, and visual contrast images.

## PRISMA Data Workflow

The detailed PRISMA workflow has been moved to:

```text
docs/prisma_workflow.md
```

The PRISMA workflow includes:

1. Converting PRISMA HE5 data to ENVI format.
2. Generating unit methane absorption coefficients using libRadtran.
3. Generating a PRISMA-specific methane target spectrum.
4. Running MAG1C to produce methane enhancement results.

The files related to PRISMA processing are stored in:

```text
prisma/
│
├── convert_prisma.R
├── prisma.ipynb
└── prisma.sh
```

## Sentinel-2 Data Workflow

The Sentinel-2 workflow uses a modified MAG1C-style script for multispectral methane detection.

Run:

```bash
cd /mnt/d/Project4/code/mag1c/mag1c
```

```bash
python mag1c_s2.py \
  ../../res/case1/20240409T074435_s2 \
  --out ../../res/case1/20240409T074435_s2_mf \
  -o
```

The output will be saved as:

```text
../../res/case1/20240409T074435_s2_mf
```

## Example Cases

### EMIT Example Case

The EMIT example case used in this repository is based on the acquisition:

```text
EMIT_L1B_RAD_001_20240409T074435_2410005_007.nc
EMIT_L1B_OBS_001_20240409T074435_2410005_007.nc
```

The corresponding processing outputs are named using the acquisition timestamp:

```text
20240409T074435_emit_mf
20240409T074435_s2_mf
```

### PRISMA Example Case

For PRISMA, the input data are first converted from HE5 to ENVI format. The converted ENVI file and its `.hdr` file are then used together with a PRISMA-specific methane target spectrum generated from libRadtran simulations.

Please see the detailed PRISMA instructions in:

```text
docs/prisma_workflow.md
```

## Notes on Data

Raw EMIT, PRISMA, and Sentinel-2 data are not included in this repository because of file size limitations. Users should download or prepare the required data separately and update the paths in the scripts accordingly.

Recommended local data structure for EMIT and Sentinel-2 processing:

```text
/mnt/d/Project4/
│
├── EMIT_data/
│   ├── EMIT_L1B_OBS_001_20240409T074435_2410005_007.nc
│   └── EMIT_L1B_RAD_001_20240409T074435_2410005_007.nc
│
└── code/
    ├── emit-utils/
    ├── mag1c/
    └── res/
```

Recommended local data structure for PRISMA processing:

```text
/mnt/d/Project2/
│
└── code/
    ├── mag1c/
    │   ├── convert_prisma.R
    │   ├── run_ch4_absorption.py
    │   ├── prisma_swir_ch4.in
    │   ├── prisma_swir_noCH4.in
    │   ├── prisma_all_ch4.in
    │   └── prisma_all_noCH4.in
    │
    └── libRadtran-2.0.6/
        └── bin/
            └── uvspec
```

## Citation

If this repository is used in research, please cite the original MAG1C paper and repository:

Foote, M. D., Dennison, P. E., Thorpe, A. K., Thompson, D. R., Jongaramrungruang, S., Frankenberg, C., & Joshi, S. C. (2020). Fast and Accurate Retrieval of Methane Concentration from Imaging Spectrometer Data Using Sparsity Prior. *IEEE Transactions on Geoscience and Remote Sensing*.

Original MAG1C repository:

https://github.com/markusfoote/mag1c

Please also refer to the EMIT SDS repository guide if using EMIT utility tools:

https://github.com/emit-sds/emit-main/wiki/Repository-Guide

For PRISMA HE5-to-ENVI conversion using `prismaread`, please refer to the original package repository:

https://github.com/lbusett/prismaread

For radiative transfer simulation using libRadtran, please refer to:

https://www.libradtran.org

## Acknowledgements

This project uses and adapts workflows based on the MAG1C methane retrieval algorithm, the EMIT utility tools, PRISMA data conversion utilities, and libRadtran radiative transfer simulations. We thank the original MAG1C developers, the EMIT SDS team, the PRISMA data/tool developers, and the libRadtran developers for making their tools and documentation publicly available.