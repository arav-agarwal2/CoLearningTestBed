
I=(1 1 1 1 1)
J=(0 1 1 1 1)
K=(0 0 1 1 1)
M=(0 0 0 1 1)
N=(0 0 0 0 1)

# for METHOD in early_fusion
# do
#     for i in ${!I[@]}
#     do
#         python synthetic/${METHOD}.py --data-path /home/yuncheng/MultiBench/synthetic/SIMPLE_DATA_CLASS=16_DIM=4_STD=0.5.pickle --input-dim 4 --hidden-dim 32 --num-classes 16 --saved-model /home/yuncheng/CLASS=16_DIM=4_STD=0.5_${METHOD}_best.pt --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/16${METHOD}_$i.txt
#     done
# done

# for METHOD in late_fusion lrf tf
# do
#     for i in ${!I[@]}
#     do
#         python synthetic/${METHOD}.py --data-path /home/yuncheng/MultiBench/synthetic/SIMPLE_DATA_CLASS=16_DIM=4_STD=0.5.pickle --input-dim 4 --output-dim 16 --hidden-dim 32 --num-classes 16 --saved-model /home/yuncheng/CLASS=16_DIM=4_STD=0.5_${METHOD}_best.pt --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/16${METHOD}_$i.txt
#     done
# done

METHOD=tf

for i in ${!I[@]}
do
    python synthetic/${METHOD}.py --data-path /home/yuncheng/MultiBench/synthetic/SIMPLE_DATA_CLASS=16_DIM=4_STD=0.5.pickle --input-dim 4 --output-dim 16 --hidden-dim 32 --num-classes 16 --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/16${METHOD}_$i.txt
done