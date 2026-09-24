#!/bin/bash

# ===== USER CONFIG =====
WRF_DIR=/path/to/WRF/run
WPS_DIR=/path/to/WPS

cd $WRF_DIR || exit

echo "===== Link met_em files ====="
ln -sf $WPS_DIR/met_em* .

echo "===== Run real.exe ====="
./real.exe || exit # Input:met_em.d01*, Output: wrfinput_d01,wrfbdy_d01

echo "===== Run wrf.exe ====="
./wrf.exe || exit # Output: wrfout_d01_2010-07-14_00:00:00, wrfout_d01_2010-07-17_00:00:00

echo "===== WRF DONE ====="

echo "PREP-CHEM-SRC-1.5"
./prep_chem_sources_RADM_WRF_FIM.exe < prep_chem_sources.inp #(Input: prep_chem_sources.inp, Output: *-g1-ab.bin, *-g1-bb.bin, *-g1-gocartBG.bin)

echo "===== WRFV3.6 ====="
./convert_emiss.exe #(Input: wrfinput_d01,wrfbdy_d01, namelist.input, *-g1-ab.bin, *-g1-bb.bin, *-g1-gocartBG.bin; Output: wrfchemi_d01,wrffirechemi_d01,wrfchemi_gocart_bg_d01)
./real.exe #(Input: met_em.d01*, Output: wrfinput_d01)
./wrf.exe
ncdiff -v SWDOWN,RAINC,T2 wrfout_d01_2010-07-14_00:00:00 wrfout_d01_2010-07-14_save diff.out