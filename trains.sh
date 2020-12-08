#!/bin/bash

nohup python3 first_model.py --hidden_size 32 --batch_size 64 > logs/c100_t10_b64_h32_e10.log &
nohup python3 first_model.py --hidden_size 32 --batch_size 128 > logs/c100_t10_b128_h32_e10.log &
nohup python3 first_model.py --hidden_size 64 --batch_size 64 > logs/c100_t10_b64_h64_e10.log &
nohup python3 first_model.py --hidden_size 64 --batch_qsize 128 > logs/c100_t10_b128_h64_e10.log &



nohup python3 first_model.py --hidden_size 32 --batch_size 64 --epochs 20 > logs/c100_t10_b64_h32_e20.log &
nohup python3 first_model.py --hidden_size 32 --batch_size 128 --epochs 20 > logs/c100_t10_b128_h32_e20.log &
nohup python3 first_model.py --hidden_size 32 --batch_size 256 --epochs 20 > logs/c100_t10_b256_h32_e20.log &
nohup python3 first_model.py --hidden_size 64 --batch_size 64 --epochs 20 > logs/c100_t10_b64_h64_e20.log &
nohup python3 first_model.py --hidden_size 64 --batch_size 128 --epochs 20 > logs/c100_t10_b128_h64_e20.log &
nohup python3 first_model.py --hidden_size 64 --batch_size 256 --epochs 20 > logs/c100_t10_b256_h64_e20.log &
nohup python3 first_model.py --hidden_size 128 --batch_size 64 --epochs 20 > logs/c100_t10_b64_h128_e20.log &
nohup python3 first_model.py --hidden_size 128 --batch_size 128 --epochs 20 > logs/c100_t10_b128_h128_e20.log &
nohup python3 first_model.py --hidden_size 128 --batch_size 256 --epochs 20 > logs/c100_t10_b256_h128_e20.log &