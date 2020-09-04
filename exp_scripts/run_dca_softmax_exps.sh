# dim=3 tau=8;emb=304;
cd ..
dataset='SHL'
for i in $(seq 0 1 1); do
  path="./results/${dataset}_dca_softmax_exp${i}/"
  mkdir -p ${path}
  export RES_PATH=${path}
  python ./trajectory_segmentation_and_features_extraction.py --trjs_path ./data/${dataset}_extracted/trjs_train.npy --labels_path ./data/${dataset}_extracted/labels_train.npy --seg_size 200 --data_type train --save_dir ./data/${dataset}_features
  python ./trajectory_segmentation_and_features_extraction.py --trjs_path ./data/${dataset}_extracted/trjs_test.npy --labels_path ./data/${dataset}_extracted/labels_test.npy --seg_size 200 --data_type test --save_dir ./data/${dataset}_features
  python ./MF_RP_mat_h5support.py --dim 3 --tau 8 --multi_feature_segs_path ./data/${dataset}_features/multi_feature_segs_train.npy --save_path ./data/${dataset}_features/multi_channel_RP_mats_train.h5
  python ./MF_RP_mat_h5support.py --dim 3 --tau 8 --multi_feature_segs_path ./data/${dataset}_features/multi_feature_segs_test.npy --save_path ./data/${dataset}_features/multi_channel_RP_mats_test.h5
  python ./PEDCC.py --save_dir ./data/${dataset}_features --dim 304
  python network_training.py --dataset ${dataset} --results-path ${path}  --RP-emb-dim 152 --FS-emb-dim 152 --patience 15 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 230 --network Dual_CA_Softmax
done