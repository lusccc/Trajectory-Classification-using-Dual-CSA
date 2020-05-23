#!/bin/bash
percentage=(0.2 0.4 0.6 0.8)


for i in $(seq 0 1 4)
do
  rm -rf ./data/geolife_features
  path="./results/shrink${percentage[${i}]}/"
  mkdir -p "${path}"
  python trajectory_segmentation_and_features.py --shrink_dataset True --shrink_ratio "${percentage[${i}]}"
  python MF_RP_mat.py
  python PEDCC.py
  python dataset_generation.py
  python Dual_SAE.py --results_path "${path}"
done


