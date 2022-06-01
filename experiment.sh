#!/bin/bash

MOSI=/usr0/home/yuncheng/MultiBench/data/mosi_raw.pkl
MOSEI=/usr0/home/yuncheng/MultiBench/data/mosei_raw.pkl
SARCASM=/usr0/home/yuncheng/MultiBench/data/sarcasm.pkl
HUMOR=/usr0/home/yuncheng/MultiBench/data/humor.pkl
AVMNIST=/usr0/home/yuncheng/MultiBench/data/avmnist

# MOSI experiments
DATASET=mosi

for METHOD in late_fusion early_fusion lf_transformer ef_transformer lrf mfm tf
do
    for MODALITIES in 0 1 2 0,1 0,2 1,2 0,1,2
    do
        python3 examples/affect/affect_${METHOD}.py --modalities [${MODALITIES}] --dataset ${DATASET} --dataset-path ${MOSI} > ${DATASET}_logs/${DATASET}_${METHOD}_${MODALITIES}.txt
    done
done 

# AVMNIST experiments
# DATASET=avmnist

# for METHOD in mfm
# do
#     for MODALITIES in 0 1 0,1
#     do
#         python3 examples/${METHOD}.py --modalities [${MODALITIES}] --dataset ${DATASET} --dataset-path ${AVMNIST} > ${DATASET}_logs/${DATASET}_${METHOD}_${MODALITIES}.txt
#     done
# done 

# DATASET=sarcasm

# for METHOD in early_fusion
# do
#     for MODALITIES in 0 1 2 0,1 0,2 1,2 0,1,2
#     do
#         python3 examples/affect/affect_${METHOD}.py --modalities [${MODALITIES}] --dataset ${DATASET} --dataset-path ${SARCASM} > ${DATASET}_${METHOD}_${MODALITIES}.txt
#     done
# done 

# Test
# METHOD=ef_transformer
# MODALITIES=0

# python3 examples/mfm.py --modalities [0] --dataset avmnist --dataset-path /usr0/home/yuncheng/MultiBench/data/avmnist
