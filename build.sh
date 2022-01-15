#!/bin/bash

# Do not use any optimization
gcc Gemm_Opt_row_major.c utils.c main.c -O3 -DOPT_LEVEL_3 -mavx -o Gemm

# Execute the program
./Gemm

# Delete program
rm -rf Gemm
