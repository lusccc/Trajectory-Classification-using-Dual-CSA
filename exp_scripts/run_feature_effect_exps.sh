# dim=3 tau=8;emb=304;
cd ..
dataset='geolife'
features=("3" "4" "7" "3,4" "3,7" "4,7" "3,4,7" "3,4,8" "3,4,7,8" "3,4,7,8,9")
for i in $(seq 7 1 8); do
  path="./results/${dataset}_feature${features[${i}]}_exp_correct"
  mkdir -p ${path}
  export RES_PATH=${path}
  rm -rf ./data/${dataset}_features/*
  python ./trajectory_segmentation_and_features_extraction.py --trjs_path ./data/${dataset}_extracted/trjs_train.npy --labels_path ./data/${dataset}_extracted/labels_train.npy --seg_size 200 --data_type train --save_dir ./data/${dataset}_features --feature_set ${features[${i}]}
  python ./trajectory_segmentation_and_features_extraction.py --trjs_path ./data/${dataset}_extracted/trjs_test.npy --labels_path ./data/${dataset}_extracted/labels_test.npy --seg_size 200 --data_type test --save_dir ./data/${dataset}_features --feature_set ${features[${i}]}
  python ./MF_RP_mat_h5support.py --dim 3 --tau 8 --multi_feature_segs_path ./data/${dataset}_features/multi_feature_segs_train.npy --save_path ./data/${dataset}_features/multi_channel_RP_mats_train.h5
  python ./MF_RP_mat_h5support.py --dim 3 --tau 8 --multi_feature_segs_path ./data/${dataset}_features/multi_feature_segs_test.npy --save_path ./data/${dataset}_features/multi_channel_RP_mats_test.h5
  python ./PEDCC.py --save_dir ./data/${dataset}_features --dim 304

  if [ ${i} -le 2 ]; then
    n_features=1
  elif [ ${i} -ge 3 ] && [ ${i} -le 5 ]; then
    n_features=2
  elif [ ${i} -eq 6 ] || [ ${i} -eq 7 ]; then
    n_features=3
  elif [ ${i} -eq 8 ]; then
    n_features=4
  elif [ ${i} -eq 9 ]; then
    n_features=5
  fi
#  python network_training.py --dataset ${dataset} --results-path ${path} --alpha 1 --beta 11 --gamma 1 --RP-emb-dim 152 --FS-emb-dim 152 --patience 20 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 330 --n-features ${n_features} --visualize-emb 0
  python network_training.py --dataset ${dataset} --results-path ${path}  --RP-emb-dim 152 --FS-emb-dim 152 --patience 15 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed  --world-size 1 --rank 0 -b 230 --n-features ${n_features}

done
