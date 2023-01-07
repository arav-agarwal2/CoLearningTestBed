
I=(1 1 1 1 1)
J=(0 1 1 1 1)
K=(0 0 1 1 1)
M=(0 0 0 1 1)
N=(0 0 0 0 1)

for METHOD in early_fusion
do
    for i in ${!I[@]}
    do
        python synthetic/${METHOD}.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/REDUNDANT_CONCEPTS_128.pickle --input-dim 32 --hidden-dim 256 --num-classes 128 --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/128${METHOD}_$i.txt
    done
done

for METHOD in late_fusion lrf
do
    for i in ${!I[@]}
    do
        python synthetic/${METHOD}.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/REDUNDANT_CONCEPTS_128.pickle --input-dim 32 --hidden-dim 256 --num-classes 128 --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/128${METHOD}_$i.txt
    done
done

for i in ${!I[@]}
do
    python synthetic/DCCA.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/REDUNDANT_CONCEPTS_128.pickle --input-dim 32 --hidden-dim 256 --num-classes 128 --saved-model/home/yuncheng/CLASS=128_STD=0.5_DCCA_best.pt --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/64DCCA_$i.txt
done

I=(1 1 1)
J=(0 1 1)
K=(0 0 1)

METHOD=tf
for i in ${!I[@]}
do
    python synthetic/${METHOD}.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/REDUNDANT_CONCEPTS_128.pickle --input-dim 32 --hidden-dim 256 --hidden-dim 256 --num-classes 128 --modalities ${I[i]} ${J[i]} ${K[i]} > synthetic/128${METHOD}_$i.txt
done

