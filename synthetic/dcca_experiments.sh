I=(1 1 1 1 1)
J=(0 1 1 1 1)
K=(0 0 1 1 1)
M=(0 0 0 1 1)
N=(0 0 0 0 1)

for i in ${!I[@]}
do
    python synthetic/DCCA.py --data-path /home/yuncheng/MultiBench/synthetic/SIMPLE_DATA_CLASS=2_DIM=1_STD=0.5.pickle --keys a b c d e label --bs 128 --input-dim 1 --output-dim 16 --hidden-dim 32 --num-classes 2 --saved-model /home/yuncheng/CLASS=2_DIM=1_STD=0.5_DCCA_best.pt --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/2DCCA_$i.txt
done

for i in ${!I[@]}
do
    python synthetic/DCCA.py --data-path /home/yuncheng/MultiBench/synthetic/SIMPLE_DATA_CLASS=16_DIM=4_STD=0.5.pickle --input-dim 4 --hidden-dim 32 --num-classes 16 --keys a b c d e label --bs 128 --saved-model /home/yuncheng/CLASS=2_DIM=1_STD=0.5_DCCA_best.pt --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/16DCCA_$i.txt
done

for i in ${!I[@]}
do
    python synthetic/DCCA.py --data-path /home/yuncheng/MultiBench/synthetic/SIMPLE_DATA_CLASS=32_DIM=5_STD=0.5.pickle --input-dim 5 --hidden-dim 64 --num-classes 32 --keys a b c d e label --bs 128 --saved-model /home/yuncheng/CLASS=2_DIM=1_STD=0.5_DCCA_best.pt --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/32DCCA_$i.txt
done

for i in ${!I[@]}
do
    python synthetic/DCCA.py --data-path /home/yuncheng/MultiBench/synthetic/SIMPLE_DATA_CLASS=64_DIM=6_STD=0.5.pickle --input-dim 6 --hidden-dim 128 --num-classes 64 --keys a b c d e label --bs 128 --saved-model /home/yuncheng/CLASS=2_DIM=1_STD=0.5_DCCA_best.pt --modalities ${I[i]} ${J[i]} ${K[i]} ${M[i]} ${N[i]} > synthetic/64DCCA_$i.txt
done
