
# for SEP in 0.01 0.02 0.05 0.1 0.2
# do
#     python synthetic/early_fusion.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/DOMINANT_${SEP}_CONCEPTS_64.pickle --input-dim 32 --hidden-dim 128 --num-classes 64 --modalities 1 1 > synthetic/64early_fusion_sep${SEP}.txt
# done

# for METHOD in late_fusion lrf
# do
#     for SEP in 0.01 0.02 0.05 0.1 0.2
#     do
#         python synthetic/${METHOD}.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/DOMINANT_${SEP}_CONCEPTS_64.pickle --input-dim 32 --output-dim 128 --hidden-dim 128 --num-classes 64 --modalities 1 1 > synthetic/64${METHOD}_sep${SEP}.txt
#     done
# done

# for SEP in 0.05 0.1 0.2
# do
#     python synthetic/DCCA.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/DOMINANT_${SEP}_CONCEPTS_64.pickle --input-dim 32 --output-dim 128 --hidden-dim 128 --num-classes 64 --saved-model /home/yuncheng/CLASS=64_DCCA_best.pt --modalities 1 1 --epochs 1 --lr 1e-4 --keys a b label > synthetic/64DCCA_sep${SEP}.txt
# done

# for METHOD in tf
# do
#     for SEP in 0.01 0.02 0.05 0.1 0.2
#     do
#         python synthetic/${METHOD}.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/DOMINANT_${SEP}_CONCEPTS_64.pickle --input-dim 32 --output-dim 128 --hidden-dim 128 --num-classes 64 --modalities 1 1 > synthetic/64${METHOD}_sep${SEP}.txt
#     done
# done

for SEP in 0.01 0.02 0.05 0.1 0.2
do
    python synthetic/InfoNCECoordination.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/DOMINANT_${SEP}_CONCEPTS_64.pickle  --keys a b label --input-dim 32 --output-dim 128 --hidden-dim 128 --num-classes 64  --saved-model /home/yuncheng/CLASS=64_SEP=${SEP}_infonce_best.pt --modalities 1 1 > synthetic/64infonce_sep${SEP}.txt
done
