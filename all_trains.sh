#!/bin/bash

epochs=20

technical_indicators=(False True)

seeds=(1 2 3 4 5)
#seeds=(1)

hidden=(32 64 128)
batch=(256 512 1024)

lr=(0.001 0.0001)
adjust_lr=(8 4)


for t in "${technical_indicators[@]}"; do
	for ((i=0;i<${#lr[@]};++i)); do
		for h in "${hidden[@]}"; do
			for b in "${batch[@]}"; do
				for s in "${seeds[@]}"; do
					modelname="${t}_s${s}_lr${lr[i]}_a${adjust_lr[i]}_h${h}_b${b}_e${epochs}"
					python3 first_model.py --modelname ${modelname}  --technical_indicators ${t} --seed ${s} --lr ${lr[i]} --adjust_lr ${adjust_lr[i]} --hidden_size $h --batch_size $b --epochs $epochs > all_logs/"${modelname}.log"
				done
			done
		done
	done
done
