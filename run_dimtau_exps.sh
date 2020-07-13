#!/bin/bash
dims=(5 6 9 9 9 9)
taus=(8 8 6 7 8 9)
# other parameters fixed to: EMB_DIM = 96,
# alpha, beta, gamma = 1, 4, 1
# features_set = 3,4,7,8,9
#MAX_SEGMENT_SIZE = 184
#MIN_N_POINTS = 10

for i in `seq 0 1 7`
do
  path="./results/dim${dims[${i}]}tau${taus[${i}]}/"
  mkdir -p ${path}
#  cp ./results/default/ts_conv_ae_check_point.model ${path}
#  cp ./results/default/RP_conv_ae_check_point.model ${path}

  python MF_RP_mat.py --dim ${dims[${i}]} --tau ${taus[${i}]} --trjs_segs_features_path "data/geolife_features/trjs_segs_features_train.npy" --save_path "data/geolife_features/RP_mats_train.npy"
  python MF_RP_mat.py --dim ${dims[${i}]} --tau ${taus[${i}]} --trjs_segs_features_path "data/geolife_features/trjs_segs_features_test.npy" --save_path "data/geolife_features/RP_mats_test.npy"

  python PEDCC.py --dataset geolife --dim 96
  python Dual_SAE.py --dataset geolife --results_path ${path}
done


