# dim=3 tau=8;emb=304;beta=11
cd ..
for i in $(seq 3 1 10); do
  path="./results/dca_softmax_SHL_exp${i}/"
  mkdir -p ${path}
  export RES_PATH=${path}
  #python ./trajectory_segmentation_and_features_extraction.py --trjs_path ./data/SHL_extracted/trjs_train.npy --labels_path ./data/SHL_extracted/labels_train.npy --seg_size 200 --data_type train --save_dir ./data/SHL_features
  #python ./trajectory_segmentation_and_features_extraction.py --trjs_path ./data/SHL_extracted/trjs_test.npy --labels_path ./data/SHL_extracted/labels_test.npy --seg_size 200 --data_type test --save_dir ./data/SHL_features
  #python ./MF_RP_mat_h5support.py --dim 3 --tau 8 --multi_feature_segs_path ./data/SHL_features/multi_feature_segs_train.npy --save_path ./data/SHL_features/multi_channel_RP_mats_train.h5
  #python ./MF_RP_mat_h5support.py --dim 3 --tau 8 --multi_feature_segs_path ./data/SHL_features/multi_feature_segs_test.npy --save_path ./data/SHL_features/multi_channel_RP_mats_test.h5
  #python ./PEDCC.py --save_dir ./data/SHL_features --dim 304
  python network_training.py --dataset SHL --results-path ${path} --alpha 1 --beta 11 --gamma 1 --RP-emb-dim 152 --FS-emb-dim 152 --patience 20 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --gpu 0 --world-size 1 --rank 0 -b 160 --network Dual_CA_Softmax
done