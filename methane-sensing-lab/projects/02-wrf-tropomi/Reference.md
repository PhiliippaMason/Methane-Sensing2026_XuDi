# References:
1. Install Gfortran: https://fortran-lang.org/en/learn/os_setup/install_gfortran/
2. perl: https://learn.perl.org/installing/windows.html
3. csh: https://www.reddit.com/r/git/comments/cq5k3x/how_to_run_a_c_shell_csh_file_on_windows/
4. WRF 
- Website: https://www2.mmm.ucar.edu/wrf/users/download/get_source.html
- Install: https://www2.mmm.ucar.edu/wrf/users/download/get_sources_new.php
- Compile: https://www2.mmm.ucar.edu/wrf/OnLineTutorial/compilation_tutorial.php
5. WRF-CHEM
- Compile: https://ruc.noaa.gov/wrf/wrf-chem/wrf_tutorial_exercises_v35/compiling_code.html
- Examples: https://ruc.noaa.gov/wrf/wrf-chem/tutorialexercises.htm
- Example-based data: https://www2.mmm.ucar.edu/wrf/users/download/get_sources_wps_geog.html
6. Visualization
- ncl: https://www.ncl.ucar.edu/Download/
7. versions:
https://www2.mmm.ucar.edu/wrf/users/download/get_sources.html#WRF-Chem
8. MEGAN INSTALL: https://zhuanlan.zhihu.com/p/362264622
9. ioapi
- https://www.cmascenter.org/ioapi/documentation/all_versions/html/AVAIL.html#build
- https://github.com/USEPA/CMAQ/blob/main/DOCS/Users_Guide/Tutorials/CMAQ_UG_tutorial_build_library_intel.md
- https://forums.developer.nvidia.com/t/megan-software-compilation/133760
10. WRF-CHEM User guide
- https://ruc.noaa.gov/wrf/wrf-chem/Users_guide.pdf
- https://ruc.noaa.gov/wrf/wrf-chem/wrf_tutorial_2018/WRF_Chem_running.pdf
11. Tutorial:
- https://ruc.noaa.gov/wrf/wrf-chem/tutorialexercises.htm
- https://ruc.noaa.gov/wrf/wrf-chem/tutorial2018.htm
12. Exp1: 

Dust: use geographical input mandatory static data + meteorological data for dust emission
- https://ruc.noaa.gov/wrf/wrf-chem/tutorialexercises/tutorialexercises001.html
- file:///D:/Project1/paper/4_dust_volcano.pdf
13. Exp2 (biogenic data + chemistry data)

Volcanic Ash: using the prep_chem tool to conduct volcanic ash simulation-global volcanic dataset

PREP model:
http://ftp.cptec.inpe.br/pesquisa/bramsrd/BRAMS_5.4/BRAMS/documentation/guide-PREP-CHEM-SRC-1.8.3.pdf

Compile Issues:
- https://forum.mmm.ucar.edu/threads/error-compiling-prep_chem_src.12039/
- https://forum.mmm.ucar.edu/threads/error-in-make-prep_chem_sources-version-1-5.12194/
- https://forum.mmm.ucar.edu/threads/error-compiling-prep_chem_src.12039/page-2

Convert_emiss.exe
https://forum.mmm.ucar.edu/threads/convert_emiss-f-in-wrfchem-v4-0.341/

14. Exp3

US Emission: generate anthropogenic emissions

Issues:
- ifort: compiled the program
- Emiss_v04_CONUS60km.exe: not run
- https://forum.mmm.ucar.edu/threads/issue-with-v3-v4-data-for-wrf4-2-1-and-ndown-exe.10197/
- https://www2.acom.ucar.edu/wrf-chem/wrf-chem-tools-community
- see "anthro_emiss" and "EPA_ANTHRO_EMIS"

15. ifort Install
https://stackoverflow.com/questions/65570841/install-ifort-on-ubuntu-20

16. Other transfer tool install (ANTHRO tool)
- https://forum.mmm.ucar.edu/threads/exercise-3-of-wrf-chem.10594/
- https://community.intel.com/t5/Intel-Fortran-Compiler/error-loading-plugin-libimf-so/td-p/1411052
- https://stackoverflow.com/questions/70687930/intel-oneapi-2022-libimf-so-no-such-file-or-directory-during-openmpi-compila
- https://community.intel.com/t5/Intel-Fortran-Compiler/Fortran-compiler-with-option-static-and-NetCDF-library/td-p/1156087

17. Exp4
/mnt/d/Project1/code/PREP-CHEM-SRC-1.5/bin

18. Exp5
https://ruc.noaa.gov/wrf/wrf-chem/wrf_tutorial_emissions_v35/exercise_2.html

19. BC
- https://www.acom.ucar.edu/wrf-chem/download.shtml `MOZBC`
- https://www.acom.ucar.edu/cam-chem/cam-chem.shtml `Boundary field data`
- https://blog.csdn.net/qq_27984679/article/details/124360384 `Tutorial`
- https://ruc.noaa.gov/wrf/wrf-chem/wrf_tutorial_nepal/talks/BCs.pdf `Official Guide`
- https://www2.acom.ucar.edu/sites/default/files/documents/CESM-WRFchem_aerosols_20190822.pdf `Parameter setting tutorial`

20. Run WRF_CHEM model using CAMS data as initial
- https://confluence.ecmwf.int/pages/viewpage.action?pageId=174865233