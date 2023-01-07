
# for SEP in 0.01 0.02 0.05 0.1 0.2
# do
#     python synthetic/early_fusion.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/DOMINANT_${SEP}_CONCEPTS_16.pickle --input-dim 32 --hidden-dim 64 --num-classes 64 --modalities 1 1 > synthetic/16early_fusion_sep${SEP}.txt
# done

# for METHOD in late_fusion lrf
# do
#     for SEP in 0.01 0.02 0.05 0.1 0.2
#     do
#         python synthetic/${METHOD}.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/DOMINANT_${SEP}_CONCEPTS_16.pickle --input-dim 32 --output-dim 64 --hidden-dim 64 --num-classes 16 --modalities 1 1 > synthetic/16${METHOD}_sep${SEP}.txt
#     done
# done

# for SEP in 0.01 0.02 0.05 0.1 0.2
# do
#     python synthetic/DCCA.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/DOMINANT_${SEP}_CONCEPTS_16.pickle --input-dim 32 --output-dim 64 --hidden-dim 64 --num-classes 16 --saved-model /home/yuncheng/CLASS=16_STD=0.5_DCCA_best.pt --modalities 1 1 --epochs 3 > synthetic/16DCCA_sep${SEP}.txt
# done

# for METHOD in tf
# do
#     for SEP in 0.01 0.02 0.05 0.1 0.2
#     do
#         python synthetic/${METHOD}.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/DOMINANT_${SEP}_CONCEPTS_16.pickle --input-dim 32 --output-dim 64 --hidden-dim 64 --num-classes 16 --modalities 1 1 > synthetic/16${METHOD}_sep${SEP}.txt
#     done
# done

for SEP in 0.01 0.02 0.05 0.1 0.2
do
    python synthetic/InfoNCECoordination.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/DOMINANT_${SEP}_CONCEPTS_16.pickle  --keys a b label --input-dim 32 --output-dim 64 --hidden-dim 64 --num-classes 16 --saved-model /home/yuncheng/CLASS=16_SEP=${SEP}_infonce_best.pt --modalities 1 1 > synthetic/16infonce_sep${SEP}.txt
done
