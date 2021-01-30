#!/bin/bash

python3 first_model_indicators_lr00001_TH.py --hidden_size 32 --batch_size 256 --epochs 20 > logs/indicators_lr0.0001_TH_c100_t10_b256_h32_e20.log
python3 first_model_indicators_lr00001_TH.py --hidden_size 32 --batch_size 512 --epochs 20 > logs/indicators_lr0.0001_TH_c100_t10_b512_h32_e20.log
python3 first_model_indicators_lr00001_TH.py --hidden_size 32 --batch_size 1024 --epochs 20 > logs/indicators_lr0.0001_TH_c100_t10_b1024_h32_e20.log
python3 first_model_indicators_lr00001_TH.py --hidden_size 64 --batch_size 256 --epochs 20 > logs/indicators_lr0.0001_TH_c100_t10_b256_h64_e20.log
python3 first_model_indicators_lr00001_TH.py --hidden_size 64 --batch_size 512 --epochs 20 > logs/indicators_lr0.0001_TH_c100_t10_b512_h64_e20.log
python3 first_model_indicators_lr00001_TH.py --hidden_size 64 --batch_size 1024 --epochs 20 > logs/indicators_lr0.0001_TH_c100_t10_b1024_h64_e20.log
python3 first_model_indicators_lr00001_TH.py --hidden_size 128 --batch_size 256 --epochs 20 > logs/indicators_lr0.0001_TH_c100_t10_b256_h128_e20.log
python3 first_model_indicators_lr00001_TH.py --hidden_size 128 --batch_size 512 --epochs 20 > logs/indicators_lr0.0001_TH_c100_t10_b512_h128_e20.log
python3 first_model_indicators_lr00001_TH.py --hidden_size 128 --batch_size 1024 --epochs 20 > logs/indicators_lr0.0001_TH_c100_t10_b1024_h128_e20.log
