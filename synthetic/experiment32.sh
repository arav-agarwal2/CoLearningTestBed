
I=(1 1 1 1 1)
J=(0 1 1 1 1)
K=(0 0 1 1 1)
M=(0 0 0 1 1)
N=(0 0 0 0 1)

for METHOD in early_fusion
do
    for i in ${!I[@]}
    do
        python synthetic/${METHOD}.py --data-path /home/yuncheng/MultiBench/synthetic/SIMPLE_DATA_CLASS=32_DIM=5_STD=0.5.pickle --input-dim 5 --hidden-dim 64 --num-classes 32 --saved-model /home/yuncheng/CLASS=32_DIM=5_STD=0.5_${METHOD}_best.pt --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/32${METHOD}_$i.txt
    done
done

for METHOD in late_fusion lrf tf
do
    for i in ${!I[@]}
    do
        python synthetic/${METHOD}.py --data-path /home/yuncheng/MultiBench/synthetic/SIMPLE_DATA_CLASS=32_DIM=5_STD=0.5.pickle --input-dim 5 --output-dim 32 --hidden-dim 64 --num-classes 32 --saved-model /home/yuncheng/CLASS=32_DIM=5_STD=0.5_${METHOD}_best.pt --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/32${METHOD}_$i.txt
    done
done