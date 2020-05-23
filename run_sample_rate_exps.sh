#!/bin/bash
percentage=(0.1 0.2 0.3 0.4 0.5 0.6)


for i in $(seq 0 1 6)
do
  rm -rf ./data/geolife_features
  path="./results/drop${percentage[${i}]}/"
  mkdir -p "${path}"
  python trajectory_segmentation_and_features.py --random_drop True --random_drop_percentage "${percentage[${i}]}"
  python MF_RP_mat.py
  python PEDCC.py
  python dataset_generation.py
  python Dual_SAE.py --results_path "${path}"
done


