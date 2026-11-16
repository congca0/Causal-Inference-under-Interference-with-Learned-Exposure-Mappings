# Spillover inference under transport-model uncertainty

Code and data for "Spillover inference when the exposure mapping is induced by an
uncertain transport model" (Cong Cao).
If you find this repository useful, please cite the accompanying paper.
Needs Python 3 with numpy, scipy, pandas and matplotlib.

    code/      scripts
    data/      merged_data.csv, county_coordinates.csv
    results/   summary tables from the simulation in the paper
    figures/   Figures 1-5

## Simulation (Sections 4-5)

    cd code
    python run_mc.py 0 100
    python analyze.py
    python make_tables.py
    python fig1_domain.py

run_mc.py saves each replicate to code/results/ and skips replicates that already
exist, so a long run can be restarted. One replicate takes 15-20 seconds on a single
core. analyze.py writes Tables 1-3 and Figures 2-5 to code/out/. Seeds are fixed.

## Kernel example (Supporting Information, Section S1)

    python kernel_example.py

## California analysis (Section 6)

    python california_reanalysis.py ../data/merged_data.csv ../data/county_coordinates.csv

Results are written to code/ca_out/.

## Data

merged_data.csv is a California county-day panel for January 2024 built from the EPA
AirData daily files (PM2.5 parameter 88101, wind speed, temperature), with monitor
values averaged within each county. county_coordinates.csv has one latitude and
longitude per county. The raw EPA files are not included. They can be downloaded from
https://aqs.epa.gov/aqsweb/airdata/download_files.html

For questions, please contact:
Cong Cao  
Department of Biostatistics  
Yale School of Public Health  
cong.cao@yale.edu
