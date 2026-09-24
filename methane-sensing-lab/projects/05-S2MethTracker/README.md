# 05-S2MethTracker

## Methodology

The methodology implemented in S2MethTracker is described in the accompanying technical documentation. The system integrates Sentinel-2 imagery, plume enhancement techniques, wind-field information, and emission-rate estimation methods to detect and quantify methane plumes.

This repository is intended as a research prototype and reproducible workflow rather than a production-ready monitoring service.

## Repository Structure

```text
S2MethTracker/
├── pipeline.ipynb   # Main workflow notebook
├── radtran.py           # Placeholder/interface for the radiative transfer model component
└── README.md        # Project documentation
```

## Project Overview

S2MethTracker provides a Sentinel-2-based workflow for methane detection and quantification. The main processing steps include:

1. Loading Sentinel-2 multispectral observations.
2. Detecting anomalous plume-like features.
3. Segmenting plumes from background.
4. Estimating methane column enhancement through a radiative transfer model component.
5. Supporting emission-rate estimation using auxiliary meteorological information.
6. Visualizing detection and quantification results.

This repository is intended as a research prototype rather than an operational methane monitoring service.

## Files

### `radtran.py`

`radtran.py` is an interface/placeholder file for the radiative transfer model component used to calculate methane column concentration estimates.

The original radiative transfer model code is **not included** in this repository.

This is because the relevant study states:

> “The radiative transfer model code used to calculate methane column concentrations will be made available upon request.”

The related study is:

**High-frequency monitoring of anomalous methane point sources with multispectral Sentinel-2 satellite observations**

Users who require the original radiative transfer model implementation should contact the authors of that study directly.

In this repository, `radtran.py` does **not** reproduce, redistribute, or claim ownership of the original radiative transfer model code. It only indicates where such a component should be integrated in the workflow.


## Documentation

A detailed description of the algorithm design, data processing workflow, and validation experiments is provided in:

- `docs/S2MethTracker.pdf`

## Disclaimer

This repository does not include the original radiative transfer model code used in the referenced study. The file `radtran.py` is provided only as a placeholder/interface for integrating such a model.

Users are responsible for obtaining permission to use the original model code from the corresponding authors of the related publication if exact reproduction is required.

This repository does not claim ownership of the original radiative transfer model code.

## Citation

If you use S2MethTracker in your research or project, please cite this repository:

```bibtex
@software{s2methtracker,
  title = {S2MethTracker: A Sentinel-2-based Methane Detection and Quantification System},
  author = {Xu, Di},
  year = {2026},
  url = {https://github.com/ICDX01/methane-sensing-lab.git}
}