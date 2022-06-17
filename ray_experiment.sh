#!/bin/bash

MOSI=/usr0/home/yuncheng/MultiBench/data/mosi_raw.pkl
MOSEI=/usr0/home/yuncheng/MultiBench/data/mosei_raw.pkl
SARCASM=/usr0/home/yuncheng/MultiBench/data/sarcasm.pkl
HUMOR=/usr0/home/yuncheng/MultiBench/data/humor.pkl
AVMNIST=/usr0/home/yuncheng/MultiBench/data/avmnist

# Affect experiments
DATASET=mosi

for METHOD in mfm
do
    for MODALITIES in 0 1 2 0,1 0,2 1,2 0,1,2
    do
        python3 ray/affect_${METHOD}.py  --modalities [${MODALITIES}] --dataset ${DATASET} --dataset-path ${MOSI} > ray/${DATASET}_logs/${DATASET}_${METHOD}_${MODALITIES}.txt
    done
done 
