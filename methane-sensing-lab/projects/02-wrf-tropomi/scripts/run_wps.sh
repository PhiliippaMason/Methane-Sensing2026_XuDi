# scripts

## scripts/run_wps.sh

```bash
#!/bin/bash

# ===== USER CONFIG =====
WPS_DIR=/path/to/WPS
DATA_DIR=/path/to/NCEP

cd $WPS_DIR || exit

echo "===== Step 1: Input NAMELIST.wps ====="
ncl util/plotgrids_new.ncl

echo "===== Step 1: geogrid ====="
./geogrid.exe || exit #output: geo_em.d01.nc
ncview geo_em.d01.nc

echo "===== Step 2: link GRIB ====="
./g2print.exe ../../../data/NCEP/2020/fnl_20200101_00_00.grib2
ln -sf ungrib/Variable_Tables/Vtable.GFS Vtable
./link_grib.csh $DATA_DIR || exit

echo "===== Step 3: ungrib ====="
./ungrib.exe || exit #output: prefix*

echo "===== Step 4: metgrid ====="
./metgrid.exe || exit #input: prefix*, output: met_em.d01*

echo "===== WPS DONE ====="

echo "Biogenic Emissions:(MEGAN)" #https://ruc.noaa.gov/wrf/wrf-chem/wrf_tutorial_emissions_v35/exercise_2.html
./megan_bio_emiss < megan_bio_emiss.inp >& run.out

echo "PRER_CHEM_SOURCES" #Change one day to one month: https://www.researchgate.net/post/How_to_provide_emissions_for_WRF-chem_using_pre-chem-src