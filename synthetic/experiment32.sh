
I=(1 1 1 1 1)
J=(0 1 1 1 1)
K=(0 0 1 1 1)
M=(0 0 0 1 1)
N=(0 0 0 0 1)

# for METHOD in early_fusion
# do
#     for i in ${!I[@]}
#     do
#         python synthetic/${METHOD}.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/REDUNDANT_CONCEPTS_32.pickle --input-dim 32 --hidden-dim 64 --num-classes 32 --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/32${METHOD}_$i.txt
#     done
# done

# for METHOD in late_fusion lrf
# do
#     for i in ${!I[@]}
#     do
#         python synthetic/${METHOD}.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/REDUNDANT_CONCEPTS_32.pickle --input-dim 32 --output-dim 64 --hidden-dim 64 --num-classes 32 --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/32${METHOD}_$i.txt
#     done
# done

# for i in ${!I[@]}
# do
#     python synthetic/DCCA.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/REDUNDANT_CONCEPTS_32.pickle --keys a b c d e label --input-dim 32 --output-dim 64 --hidden-dim 64 --num-classes 32 --saved-model /home/yuncheng/CLASS=32_STD=0.5_DCCA_best.pt --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} --epochs 4 > synthetic/32DCCA_$i.txt
# done

# for i in ${!I[@]}
# do
#     python synthetic/InfoNCECoordination.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/REDUNDANT_CONCEPTS_32.pickle --keys a b c d e label --input-dim 32 --output-dim 64 --hidden-dim 64 --num-classes 32 --saved-model /home/yuncheng/CLASS=32_STD=0.5_infonce_best.pt --lr 1e-4 --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/32infonce_${i}.txt
# done

for i in ${!I[@]}
do
    python synthetic/mctn.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/REDUNDANT_CONCEPTS_32.pickle --keys a b c d e label --input-dim 32 --hidden-dim 64 --num-classes 32 --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/32mctn_${i}.txt
done

# I=(1 1 1)
# J=(0 1 1)
# K=(0 0 1)

# METHOD=tf
# for i in ${!I[@]}
# do
#     python synthetic/${METHOD}.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/REDUNDANT_CONCEPTS_32.pickle --input-dim 32 --output-dim 64 --hidden-dim 64 --num-classes 32 --modalities ${I[i]} ${J[i]} ${K[i]} > synthetic/32${METHOD}_$i.txt
# done

