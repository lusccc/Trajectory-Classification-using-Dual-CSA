#!/bin/bash
ns=(152 152 152 152 152 152 152 152 152 152 160 160 160 160 160 160 160 160 168 168 168 168 168 168 176 176 176 176 184 184 192)
dims=(5 6 7 8 9 9 9 9 9 9 6 7 8 9 9 9 9 9 7 8 9 9 9 9 8 9 9 9 9 9 9)
taus=(8 8 8 8 4 5 6 7 8 9 8 8 8 5 6 7 8 9 8 8 6 7 8 9 8 7 8 9 8 9 9)
# other parameters fixed to: EMB_DIM = 96,
# loss_weights = 1, 3, 1
# features_set = 3,4,7,8,9
cd ..
for i in `seq 0 1 30`
do
  path="./results/rep1_seg${ns[${i}]}dim${dims[${i}]}tau${taus[${i}]}/"
  mkdir -p ${path}
  python ./trajectory_segmentation_and_features.py --trjs_path ./data/geolife_extracted/trjs_train.npy --labels_path ./data/geolife_extracted/labels_train.npy --seg_size ${ns[${i}]} --data_type train --save_dir ./data/geolife_features
  python ./trajectory_segmentation_and_features.py --trjs_path ./data/geolife_extracted/trjs_test.npy --labels_path ./data/geolife_extracted/labels_test.npy --seg_size ${ns[${i}]} --data_type test --save_dir ./data/geolife_features
  python ./MF_RP_mat.py --dim ${dims[${i}]} --tau ${taus[${i}]} --trjs_segs_features_path ./data/geolife_features/trjs_segs_features_train.npy --save_path ./data/geolife_features/RP_mats_train.npy
  python ./MF_RP_mat.py --dim ${dims[${i}]} --tau ${taus[${i}]} --trjs_segs_features_path ./data/geolife_features/trjs_segs_features_test.npy --save_path ./data/geolife_features/RP_mats_test.npy
  python ./PEDCC.py --features_path ./data/geolife_features/trjs_segs_features_train.npy  --save_dir ./data/geolife_features --data_type train --dim 96
  python ./PEDCC.py --features_path ./data/geolife_features/trjs_segs_features_test.npy  --save_dir ./data/geolife_features --data_type test --dim 96
  python -m network.Dual_SAE --dataset geolife --results_path ${path}
done


