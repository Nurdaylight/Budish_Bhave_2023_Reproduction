# Budish_Bhave_2023_Reproduction

### Replication Exercise for Computational Economics
**Authors**:MIKHAIL MARTIANOV, NURDAULET MENGLIBAYEV

---

## Overview

This is the README file for the replication of Budish & Bhave, "Primary-Market Auctions for Event Tickets: Eliminating the Rents of 'Bob the Broker'?" 2023 The original study explores auction mechanisms to improve ticket allocation efficiency and reduce broker rent extraction. The original code and data were developed in STATA and the anaysis have been replicated here using Julia.

---

## Computational Requirements

### Julia Version
- Julia 1.11.2

### Required Packages
- StatFiles, DataFrames, Statistics, GLM,  Plots,   OnlineStats, StatsBase, OnlineStats,
- CovarianceMatrices, RDatasets, FixedEffectModels, Random, Bootstrap, Pipe, Parameters, LinearAlgebra, KernelDensity

---

## Scripts

The repository contains the following Julia scripts:

1. **BB23.jl**: Contains the complete code for the analysis as the module. Including outputs of Figures and Tables. [NOTE: The authors manually costructed tables from outputs, thus we simply replicated output values]
3. **RUN.jl**: The main file that ties all scripts together and runs the replication. Contains the BB23.run() using the Main.BB23 package generated as instructed. Calling BB23.run() lets the complete output into the terminal and stores figures in respective folder. 
---

## Instructions for Replication

### Step-by-Step Guide

1. Download complete repository found at :
   ```bash
   git clone https://github.com/Nurdaylight/Budish_Bhave_2023_Reproduction/
   ```

2. Navigate and edit the to the project directory's folder data:
   ```bash
   "<your path name>.\Data"
   ```

3. Open the Julia REPL:
   ```bash
   julia
   ```

4. Activate the project environment and install dependencies:
   ```julia
   using Pkg
   Pkg.add(["StatFiles", "DataFrames", "Statistics", "GLM", "Plots", "StatsBase",  "OnlineStats",
            "CovarianceMatrices" , "RDatasets", "FixedEffectModels",  "Random", "Bootstrap", "Pipe", "Parameters", "LinearAlgebra", "KernelDensity"])
   Pkg.instantiate()
   using Main.BB23
   BB23.run()
   
   ```
---

## Notes on Replication

### Discrepancies

While most results align with the original paper, minor discrepancies were observed.

- The main discrepancies arise in boostrapped results, since seed's of Stata and those of Julia differ significantly in draws. Reproducing exact analysis on Julia modified data using Stata estimation yielded simmilar results.The differences were small of about 1 to 2% at most. 
- There is small discrepancy in scaling of Figure 3 compared to the original, since on the stata vertical axis labels were directly imposed on the generated plot. While we tried to match the exact scaling slight discrepancy arised, however absolute values before scale application were exactly the same. 

### Outputs

- **Table Outputs**: Table values are outputted directly as individual or regression values simmilar to the original work. 
- **Figure Outputs**: Plots are saved as `.png` files in the `Data/Output` folder.

---

## Data Availability

The data used was provided in the original replication package found at https://www.aeaweb.org/articles?id=10.1257/mic.20180230  . 

---


## Additional Notes

For questions or issues, please contact MIKHAIL MARTIANOV, NURDAULET MENGLIBAYEV. Contributions and suggestions to improve the replication are welcome!

