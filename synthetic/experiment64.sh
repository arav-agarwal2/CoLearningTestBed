
I=(1 1 1 1 1)
J=(0 1 1 1 1)
K=(0 0 1 1 1)
M=(0 0 0 1 1)
N=(0 0 0 0 1)

# for METHOD in early_fusion
# do
#     for i in ${!I[@]}
#     do
#         python synthetic/${METHOD}.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/REDUNDANT_CONCEPTS_64.pickle --input-dim 32 --hidden-dim 128 --num-classes 64 --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/64${METHOD}_$i.txt
#     done
# done

# for METHOD in late_fusion lrf
# do
#     for i in ${!I[@]}
#     do
#         python synthetic/${METHOD}.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/REDUNDANT_CONCEPTS_64.pickle --input-dim 32 --hidden-dim 128 --num-classes 64 --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/64${METHOD}_$i.txt
#     done
# done

# for i in ${!I[@]}
# do
#     python synthetic/DCCA.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/REDUNDANT_CONCEPTS_64.pickle --keys a b c d e label --input-dim 32 --hidden-dim 128 --num-classes 64 --saved-model /home/yuncheng/CLASS=32_STD=0.5_DCCA_best.pt --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} --epochs 5 > synthetic/64DCCA_$i.txt
# done

# for i in ${!I[@]}
# do
#     python synthetic/InfoNCECoordination.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/REDUNDANT_CONCEPTS_64.pickle --keys a b c d e label --input-dim 32 --output-dim 128 --hidden-dim 128 --num-classes 64 --saved-model /home/yuncheng/CLASS=64_STD=0.5_infonce_best.pt --lr 1e-4 --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/64infonce_${i}.txt
# done

for i in ${!I[@]}
do
    python synthetic/mctn.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/REDUNDANT_CONCEPTS_64.pickle --keys a b c d e label --input-dim 32 --hidden-dim 128 --num-classes 64 --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/64mctn_${i}.txt
done

# I=(1 1 1)
# J=(0 1 1)
# K=(0 0 1)

# METHOD=tf
# for i in ${!I[@]}
# do
#     python synthetic/${METHOD}.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/REDUNDANT_CONCEPTS_64.pickle --input-dim 32 --hidden-dim 128 --num-classes 64 --modalities ${I[i]} ${J[i]} ${K[i]} > synthetic/64${METHOD}_$i.txt
# done

