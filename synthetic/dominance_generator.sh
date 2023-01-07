for CONCEPTS in 4 8
do
    for SEP in 0.01 0.02 0.05 0.1 0.2
    do
        python synthetic/generate.py --dominant-modality-concept-variance ${SEP} --concept-number ${CONCEPTS} --out-path /usr0/home/yuncheng/MultiBench/synthetic/DOMINANT_${SEP}_CONCEPTS_${CONCEPTS}.pickle
    done
done