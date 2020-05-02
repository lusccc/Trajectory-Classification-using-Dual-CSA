#!/bin/bash
features=("3" "4" "7" "3,4" "3,7" "4,7" "3,4,7" "3,4,7,8" "3,4,7,8,9")


for i in `seq 0 1 9`
do
  path="./results/feature${features[${i}]}/"
  mkdir -p ${path}
  python trajectory_segmentation_and_features.py --feature_set ${features[${i}]}
  python MF_RP_mat.py --feature_set ${features[${i}]}
  python dataset_generation.py
  python Dual_SAE.py --results_path ${path}
done

