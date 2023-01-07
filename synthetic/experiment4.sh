
I=(1 1 1 1 1)
J=(0 1 1 1 1)
K=(0 0 1 1 1)
M=(0 0 0 1 1)
N=(0 0 0 0 1)

# for METHOD in early_fusion
# do
#     for i in ${!I[@]}
#     do
#         python synthetic/${METHOD}.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/REDUNDANT_CONCEPTS_4.pickle --input-dim 32 --hidden-dim 64 --num-classes 4 --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/4${METHOD}_$i.txt
#     done
# done

# for METHOD in late_fusion lrf 
# do
#     for i in ${!I[@]}
#     do
#         python synthetic/${METHOD}.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/REDUNDANT_CONCEPTS_4.pickle --input-dim 32 --output-dim 64 --hidden-dim 64 --num-classes 4 --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/4${METHOD}_$i.txt
#     done
# done

# for i in ${!I[@]}
# do
#     python synthetic/DCCA.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/REDUNDANT_CONCEPTS_4.pickle --keys a b c d e label --input-dim 32 --output-dim 64 --hidden-dim 64 --num-classes 4 --saved-model /home/yuncheng/CLASS=4_DIM=1_STD=0.5_DCCA_best.pt --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/4DCCA_$i.txt
# done


# for i in ${!I[@]}
# do
#     python synthetic/InfoNCECoordination.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/REDUNDANT_CONCEPTS_4.pickle --keys a b c d e label --input-dim 32 --output-dim 64 --hidden-dim 64 --num-classes 4 --saved-model /home/yuncheng/CLASS=4_STD=0.5_infonce_best.pt --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/4infonce_${i}.txt
# done

for i in ${!I[@]}
do
    python synthetic/mctn.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/REDUNDANT_CONCEPTS_4.pickle --keys a b c d e label --input-dim 32 --hidden-dim 64 --num-classes 4 --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/4mctn_${i}.txt
done

# I=(1 1 1)
# J=(0 1 1)
# K=(0 0 1)

# for i in ${!I[@]}
# do
#     python synthetic/tf.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/REDUNDANT_CONCEPTS_4.pickle --input-dim 32 --output-dim 64 --hidden-dim 64 --num-classes 4 --modalities ${I[i]} ${J[i]} ${K[i]} > synthetic/4tf_$i.txt
# done

