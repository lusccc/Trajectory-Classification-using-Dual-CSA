cd ..
percentage=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)

for i in $(seq 0 1 7); do
  rm -rf ./data/geolife_features/*test.npy
  rm -rf ./data/geolife_features/*test.h5
  path="./results/geolife_drop${percentage[${i}]}_exp0/"
  export RES_PATH=${path}
  mkdir -p "${path}"
  cp ./results/Dual_CSA.pt  ${path}
  python ./trajectory_segmentation_and_features_extraction.py --trjs_path ./data/geolife_extracted/trjs_test.npy --labels_path ./data/geolife_extracted/labels_test.npy --seg_size 200 --data_type test --save_dir ./data/geolife_features --random_drop_percentage "${percentage[${i}]}"
  python ./MF_RP_mat_h5support.py --dim 3 --tau 8 --multi_feature_segs_path ./data/geolife_features/multi_feature_segs_test.npy --save_path ./data/geolife_features/multi_channel_RP_mats_test.h5
  python network_training.py --dataset geolife --results-path ${path}  --RP-emb-dim 152 --FS-emb-dim 152 --patience 20 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --gpu 1 --world-size 1 --rank 0 -b 230 --e
  rm -rf ${path}/Dual_CSA.pt
done
