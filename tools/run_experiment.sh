#!/bin/bash
#PBS -l walltime=30:00:00
#PBS -r n
#PBS -e $HOME/class_project/outputs/error.txt
#PBS -o $HOME/class_project/outputs/output.txt

module load python/3.5.1
module load openblas/0.2.18
module load theano/python3.5/0.7.0

cd $HOME/class_project

pip install -r requirements.txt
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python tools/test_gpu.py
