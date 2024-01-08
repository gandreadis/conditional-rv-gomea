#!/bin/bash

module purge
module load library/blas/0.3.13/gcc-8.3.1
module load library/lapack/3.9.0/gcc-8.3.1

module load system/python/3.8.1

pip3.8 install --user numpy pandas

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gandreadis/arma/lib64

make
