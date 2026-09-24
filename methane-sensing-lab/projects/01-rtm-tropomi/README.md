# 01-rtm-tropomi

This repository contains materials for working with TROPOMI data and building a Radiative Transfer Model (RTM) forward model for greenhouse gas retrieval studies.

## Repository Structure

The folder currently contains three main files:

1. **TROPOMI data reading**  
   A notebook for reading and handling TROPOMI products.

2. **HARP toolkit**  
   The HARP package is used to import, subset, and preprocess satellite products.

3. **`harp_example.ipynb`**  
   An example notebook showing how to use HARP to read TROPOMI data.

In addition, this project is intended to include the **RTM forward model** used in the radiative transfer process.

## Project Goal

The main goal of this project is to:

- read and preprocess TROPOMI satellite observations,
- understand the role of the HARP toolkit in data handling,
- build or document the RTM forward model used in retrieval workflows.

## RTM Forward Model

The forward model describes how atmospheric state variables and surface properties are mapped to top-of-atmosphere (TOA) radiance observed by the sensor.

A typical RTM forward model includes:

- atmospheric vertical pressure layers,
- gas sub-columns (such as CH4, CO, H2O, and O2),
- pressure- and temperature-dependent absorption cross sections,
- aerosol optical properties,
- observation geometry (solar zenith angle, viewing zenith angle, relative azimuth angle),
- surface albedo,
- solar reference spectrum,
- instrument spectral response function.

The general workflow is:

1. define the atmospheric pressure grid,
2. calculate gas absorption in each layer,
3. calculate aerosol scattering and extinction,
4. solve the radiative transfer equation,
5. simulate high-resolution TOA radiance,
6. convolve with the instrument spectral response,
7. generate the final simulated spectrum.

## Requirements

Typical dependencies may include:

- Python 3.x
- `numpy`
- `xarray`
- `matplotlib`
- `netCDF4`
- `harp`
- `jupyter`

## Usage

### Read TROPOMI data

Open the example notebook:

```bash
jupyter notebook harp_example.ipynb
```

### Use HARP for preprocessing

The notebook demonstrates a basic workflow for importing and processing TROPOMI data with HARP.

## Notes

- This repository is intended for research, learning, and algorithm development.
- The RTM forward model file can be expanded later with equations, pseudocode, or implementation details.
- More examples and documentation can be added as the project develops.