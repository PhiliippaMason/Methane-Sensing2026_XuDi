# WPS Notes & Experiments

This document records WPS preprocessing steps and related WRF explanations / experiments.

---

## 1. What is WPS?

**WPS (WRF Preprocessing System)** is a collection of Fortran/C programs that prepares input data for WRF real-data cases.

Core programs:

- **geogrid**: define model domain and interpolate static geographic datasets
- **ungrib**: decode GRIB1/GRIB2 meteorological data to WPS intermediate format
- **metgrid**: horizontally interpolate meteorological fields to the model grid

Key inputs:

- `namelist.wps` (main configuration)
- static geographic datasets (WPS_GEOG)
- meteorological GRIB datasets (e.g., NCEP/FNL, GFS)

Key outputs:

- `geo_em.d0*.nc`
- `FILE:*` (intermediate output from ungrib)
- `met_em.d0*.*.nc`

---

## 2. Data Preparation

### 2.1 Geographic static data (WPS_GEOG)

Download from WRF official page (WPS geographic data) and set `geog_data_path` in `namelist.wps`.

Example path (your note):

- `D:\Project1\data\geog_high_res_mandatory\WPS_GEOG`

Typical fields include terrain height, land use, soil, albedo, vegetation, etc.

---

### 2.2 Meteorological data (GRIB)

Common sources (from your notes):

- NCEP (e.g., ds083.2 / ds094.0 from UCAR RDA)
- GFS / NAM / RUC
- NCEP/NCAR Reanalysis

Example (your note):

- ds083.2: `D:\Project1\data\NCEP_83`
- ds094.0: `D:\Project1\data\NCEP_FLEXPART_model`

---

## 3. Step-by-step: WPS Workflow (Real Data Case)

> The commands below follow your recorded procedure.

### Step 0 — Configure `namelist.wps`

1. `cd $WPS_PATH`
2. Edit `namelist.wps`:
   - domain center / map projection
   - `geog_data_path`
   - `start_date`, `end_date`
   - `interval_seconds`
   - `prefix`

### Step 1 — Check domain location (recommended)

Use NCL script to plot domains:

```bash
ncl util/plotgrids_new.ncl
```

### Step 2 — Run geogrid
```bash
./geogrid.exe
```
Output:
   - geo_em.d01.nc (and nested domains if configured)

Quick check:
```bash
ncview geo_em.d01.nc
```

### Step 3 — Prepare GRIB & run ungrib

3.1 (Optional) Check GRIB variables / format
```bash
./g2print.exe /path/to/fnl.2020-01-01_00.grib2
```

3.2 Link GRIB files
```bash
./link_grib.csh /path/to/met_data_dir/
```
Outputs: GRIBFILE.*

3.3 Select the correct Vtable
```bash
ln -sf ungrib/Variable_Tables/Vtable.GFS Vtable
```

3.4 Run ungrib
```bash
./ungrib.exe
```
Outputs (intermediate files): FILE:YYYY-MM-DD_HH

### Step 4 — Run metgrid
```bash
./metgrid.exe
```
Outputs: met_em.d01.YYYY-MM-DD_HH:00:00.nc

Check:
```bash
ncview met_em.d01.YYYY-MM-DD_HH\:00\:00.nc
```

(Optional) inspect vertical levels:
```bash
ncdump -h met_em.d01.YYYY-MM-DD_HH\:00\:00.nc | tee log.ncdump
```