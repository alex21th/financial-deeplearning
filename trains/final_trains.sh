#!/bin/bash

epochs=30

seeds=(1 2 3 4 5)
#seeds=(1)

batch=(1024 2048 4096 8192 16384)
#batch=(32768 65536 131072 262144 524288)

LOGS_DIR="final_logs/"  # !!!! CHANGE IF NECESSARY !!!!

for b in "${batch[@]}"; do
  for s in "${seeds[@]}"; do
    modelname="s${s}_b${b}_e${epochs}"
    python3 model.py --modelname ${modelname} --technical_indicators True --seed ${s} --lr 0.0001 --adjust_lr 4 --hidden_size 32 --batch_size $b --epochs $epochs > ${LOGS_DIR}"${modelname}.log"
  done
done