# AI Methane Emission Quantification

An AI-based framework for quantifying methane emission rates from satellite observations.

This repository contains the implementation used for the methane emission-rate quantification component of the study. The framework combines satellite-derived methane plume information with meteorological information, particularly wind speed, to estimate methane emission rates from detected point sources.

## Features

* AI-based methane emission-rate quantification
* Integration of methane plume information and wind conditions
* Dedicated data loading and preprocessing pipeline
* Neural-network-based quantification models
* Training and testing scripts
* Wind-speed processing and analysis
* Jupyter notebooks for model evaluation and emission-rate analysis

## Project Structure

```text
09-ai-quantification/

├─ emi_rate.ipynb     # Emission-rate analysis and visualisation
├─ loader.py          # Data loading and preprocessing
├─ models.py          # Neural-network models for emission-rate quantification
├─ README.md          # Project documentation
├─ test_q.ipynb       # Model testing and quantitative evaluation
├─ train_q.py         # Model training
├─ trainer.py         # Training utilities and training procedure
├─ windspeed.ipynb    # Wind-speed analysis and visualisation
└─ windspeed.py       # Wind-speed processing and related utilities
```

## Workflow

The overall workflow consists of four main steps:

```text
Satellite methane observations
            │
            ▼
     Data preprocessing
            │
            ├───────────────┐
            ▼               ▼
 Methane plume features   Wind information
            │               │
            └───────┬───────┘
                    ▼
       AI emission-rate model
                    │
                    ▼
       Estimated methane emission rate
```

The model learns the relationship between the observed methane plume characteristics, environmental information, and the corresponding methane emission rate.

## Quick Start

### 1. Prepare the data

Prepare the input data required by the data loader and place the data in the corresponding directories expected by the code.

The data should contain the methane observation information and the associated variables required by the quantification model, including wind-related information where applicable.

### 2. Train the model

The main training script is:

```bash
python train_q.py
```

The training procedure is implemented in `trainer.py`, while the model architectures are defined in `models.py`.

### 3. Test the model

Model performance can be evaluated using:

```text
test_q.ipynb
```

The notebook provides the testing and evaluation workflow for the trained quantification model.

### 4. Analyse emission rates

The notebook

```text
emi_rate.ipynb
```

can be used to analyse and visualise the estimated methane emission rates.

## Wind-Speed Processing

Wind information is an important input for methane emission-rate quantification because the spatial distribution of a methane plume is strongly affected by atmospheric transport.

The wind-speed processing workflow is implemented in:

```text
windspeed.py
```

and can be explored through:

```text
windspeed.ipynb
```

These files provide the processing and analysis of wind-related information used by the quantification framework.

## Main Components

### `loader.py`

Provides the data-loading and preprocessing functions required by the quantification model.

It prepares the input data before they are passed to the neural network.

### `models.py`

Contains the neural-network architectures used for methane emission-rate quantification.

### `trainer.py`

Implements the model training procedure, including the optimisation process and training utilities.

### `train_q.py`

Main entry point for training the emission-rate quantification model.

### `test_q.ipynb`

Notebook for testing the trained model and evaluating its emission-rate estimation performance.

### `emi_rate.ipynb`

Notebook for analysing the estimated methane emission rates and generating relevant results and visualisations.

### `windspeed.py`

Contains the processing utilities related to wind-speed information used in the quantification workflow.

### `windspeed.ipynb`

Notebook for inspecting and analysing the wind-speed data.


## Citation

If you use this code or the associated methodology in your research, please cite the corresponding paper:

```text
Xu, D., Mason, P., Liu, J. and Wang, Y., 2025. Meqnet: Deep learning for methane point source emission quantification from sentinel-2 observations. In Proc. NeurIPS Workshop Tackling Climate Change Mach. Learn (pp. 1-6).
```

## Notes

This repository focuses on the **emission-rate quantification** stage of the methane monitoring framework. It is complementary to the methane detection and plume identification components of the overall study.

The code is intended primarily for research and reproducibility purposes.
