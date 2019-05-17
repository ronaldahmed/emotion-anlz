#!/bin/bash

cd /home/acosta/emotion-anlz/
conda activate morph

clsf=$1
nj=$2

python3 run_classifier.py --clsf $clsf --tune --njobs $nj