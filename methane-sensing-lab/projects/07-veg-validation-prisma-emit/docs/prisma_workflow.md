# PRISMA Methane Detection Workflow

This document describes the PRISMA methane detection workflow used in this repository. The workflow includes PRISMA HE5-to-ENVI conversion, methane absorption coefficient simulation using libRadtran, generation of a PRISMA-specific methane target spectrum, and MAG1C-based methane enhancement retrieval.

## Overview

PRISMA hyperspectral data are distributed in HE5 format. Before methane enhancement retrieval, the data need to be converted into ENVI format so that they can be read by the MAG1C workflow.

The overall PRISMA processing steps are:

1. Convert PRISMA HE5 data to ENVI format.
2. Generate unit methane absorption coefficients corresponding to the wavelengths in the ENVI header.
3. Generate a PRISMA-specific methane target spectrum.
4. Run MAG1C to produce methane enhancement results.
5. Inspect and visualize the results using the PRISMA notebook.

The PRISMA-related files in this repository are stored in:

```text
prisma/
│
├── convert_prisma.R
├── prisma.ipynb
└── prisma.sh
```

Additional files used for methane absorption simulation and target spectrum generation may include:

```text
prisma_swir_ch4.in
prisma_swir_noCH4.in
prisma_all_ch4.in
prisma_all_noCH4.in
run_ch4_absorption.py
```

## Step 1: Install R and prismaread

PRISMA HE5-to-ENVI conversion can be performed using ENVI software or using the provided R script:

```text
prisma/convert_prisma.R
```

R can be downloaded from:

https://cran.r-project.org

After installing R, open a command line and start R:

```bash
R
```

Install the required R package:

```r
install.packages("remotes")
remotes::install_github("lbusett/prismaread")
```

The `prismaread` package is used to read and process PRISMA products.

## Step 2: Configure HDF5 Environment Variables

Before converting PRISMA HE5 files, the HDF5 library paths may need to be configured. For example:

```bash
export LD_LIBRARY_PATH=/usr/local/hdf5-1.12.2/lib:$LD_LIBRARY_PATH
export HDF5_DIR=/usr/local/hdf5-1.12.2
export HDF5_PLUGIN_PATH=/usr/local/hdf5/lib/plugin
```

Please update these paths according to the local HDF5 installation.

## Step 3: Convert PRISMA HE5 to ENVI Format

Go to the MAG1C working directory:

```bash
cd /mnt/d/Project2/code/mag1c
```

Run the PRISMA conversion script:

```bash
Rscript convert_prisma.R
```

This step converts PRISMA HE5 data into ENVI format. The output should include an ENVI image file and a corresponding `.hdr` header file.

The wavelength information in the `.hdr` file is used in the next step to generate the methane absorption coefficient and target spectrum.

## Step 4: Generate Unit Methane Absorption Coefficients Using libRadtran

The methane target spectrum should match the PRISMA wavelength configuration. Unit methane absorption coefficients are generated using libRadtran simulations with and without methane absorption.

libRadtran can be downloaded from:

https://www.libradtran.org/doku.php?id=download

Go to the libRadtran binary directory:

```bash
cd /mnt/d/Project2/code/libRadtran-2.0.6/bin
```

Run the methane and no-methane simulations for the SWIR wavelength range:

```bash
./uvspec < /mnt/d/Project2/code/mag1c/prisma_swir_ch4.in > /mnt/d/Project2/code/mag1c/prisma_swir_ch4.out
```

```bash
./uvspec < /mnt/d/Project2/code/mag1c/prisma_swir_noCH4.in > /mnt/d/Project2/code/mag1c/prisma_swir_noCH4.out
```

Run the methane and no-methane simulations for the full wavelength range:

```bash
./uvspec < /mnt/d/Project2/code/mag1c/prisma_all_ch4.in > /mnt/d/Project2/code/mag1c/prisma_all_ch4.out
```

```bash
./uvspec < /mnt/d/Project2/code/mag1c/prisma_all_noCH4.in > /mnt/d/Project2/code/mag1c/prisma_all_noCH4.out
```

These simulations produce radiative transfer outputs with and without methane absorption. The difference between the methane and no-methane simulations is used to derive the unit methane absorption coefficient.

## Step 5: Generate the PRISMA Methane Target Spectrum

After generating the libRadtran outputs, run the target spectrum generation script:

```bash
cd /mnt/d/Project2/code/mag1c
python run_ch4_absorption.py
```

This step generates a PRISMA-specific methane target spectrum that is consistent with the wavelength configuration in the ENVI header.

## Step 6: Run MAG1C on PRISMA Data

After converting the PRISMA HE5 data to ENVI format and generating the methane target spectrum, MAG1C can be used to produce methane enhancement results.

The general command format is:

```bash
python mag1c.py \
  <PRISMA_ENVI_INPUT> \
  --out <PRISMA_MAG1C_OUTPUT> \
  -o
```

For example:

```bash
cd /mnt/d/Project2/code/mag1c/mag1c
python mag1c.py \
  ../prisma/<prisma_envi_input> \
  --out ../res/<prisma_case>_mf \
  -o
```

Please update the input and output paths according to the actual PRISMA case name and local directory structure.

## Step 7: Inspect and Visualize PRISMA Results

The notebook

```text
prisma/prisma.ipynb
```

can be used to inspect PRISMA data, check wavelength information, validate converted ENVI files, and visualize MAG1C methane enhancement results.

Typical checks include:

* Reading the converted ENVI image and `.hdr` file.
* Checking the wavelength range and band order.
* Inspecting the generated methane target spectrum.
* Visualizing the MAG1C methane enhancement map.
* Comparing plume locations with RGB or SWIR visualizations.

## Recommended Local Directory Structure

A recommended local structure for PRISMA processing is:

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

Please update all file paths in the scripts according to the actual local setup.

## Notes

Raw PRISMA HE5 data, converted ENVI products, libRadtran outputs, and MAG1C results are not included in this repository because of file size limitations.

The following file types should generally not be committed to GitHub:

```text
*.he5
*.HDF5
*.hdf5
*.out
*.dat
*.bil
*.bip
*.bsq
*.img
*.hdr
*.npy
*.npz
*.tif
*.tiff
```

## References

For PRISMA HE5-to-ENVI conversion using `prismaread`, please refer to:

https://github.com/lbusett/prismaread

For radiative transfer simulation using libRadtran, please refer to:

https://www.libradtran.org

For methane matched-filter retrieval using MAG1C, please refer to:

https://github.com/markusfoote/mag1c