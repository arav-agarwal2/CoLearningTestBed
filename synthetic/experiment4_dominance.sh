
for SEP in 0.01 0.02 0.05 0.1 0.2
do
    python synthetic/early_fusion.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/DOMINANT_${SEP}_CONCEPTS_4.pickle --input-dim 32 --hidden-dim 64 --num-classes 4 --modalities 1 1 > synthetic/4early_fusion_sep${SEP}.txt
done

for METHOD in late_fusion lrf 
do
    for SEP in 0.01 0.02 0.05 0.1 0.2
    do
        python synthetic/${METHOD}.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/DOMINANT_${SEP}_CONCEPTS_4.pickle --input-dim 32 --output-dim 64 --hidden-dim 64 --num-classes 4 --modalities 1 1 > synthetic/4${METHOD}_sep${SEP}.txt
    done
done

for SEP in 0.01 0.02 0.05 0.1 0.2
do
    python synthetic/DCCA.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/DOMINANT_${SEP}_CONCEPTS_4.pickle --keys a b label --input-dim 32 --output-dim 64 --hidden-dim 64 --num-classes 4 --saved-model /home/yuncheng/CLASS=4_SEP=${SEP}_DCCA_best.pt --modalities 1 1 > synthetic/4DCCA_sep${SEP}.txt
done


for METHOD in tf
do
    for SEP in 0.01 0.02 0.05 0.1 0.2
    do
        python synthetic/${METHOD}.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/DOMINANT_${SEP}_CONCEPTS_4.pickle --input-dim 32 --output-dim 64 --hidden-dim 64 --num-classes 4 --modalities 1 1 > synthetic/4${METHOD}_sep${SEP}.txt
    done
done

for SEP in 0.01 0.02 0.05 0.1 0.2
do
    python synthetic/InfoNCECoordination.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/DOMINANT_${SEP}_CONCEPTS_4.pickle --keys a b label --input-dim 32 --output-dim 64 --hidden-dim 64 --num-classes 4 --saved-model /home/yuncheng/CLASS=4_SEP=${SEP}_infonce_best.pt --modalities 1 1 > synthetic/4infonce_sep${SEP}.txt
done
