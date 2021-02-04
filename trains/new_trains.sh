#!/bin/bash

epochs=30

technical_indicators=(False True)

seeds=(1 2 3 4 5)
#seeds=(1)

hidden=(32 64 128)
batch=(256 512 1024)

LOGS_DIR="new_logs/"  # !!!! CHANGE IF NECESSARY !!!!

for t in "${technical_indicators[@]}"; do
  for h in "${hidden[@]}"; do
    for b in "${batch[@]}"; do
      for s in "${seeds[@]}"; do
        modelname="${t}_s${s}_h${h}_b${b}_e${epochs}"
        python3 model.py --modelname ${modelname}  --technical_indicators ${t} --seed ${s} --lr 0.000160 --adjust_lr 99 --hidden_size $h --batch_size $b --epochs $epochs > ${LOGS_DIR}"${modelname}.log"
      done
    done
  done
done
