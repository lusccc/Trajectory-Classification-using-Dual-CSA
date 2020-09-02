cd ..
results_path=./results/test_torch_geo
export RES_PATH=${results_path}
python ./trajectory_segmentation_and_features_extraction.py --trjs_path ./data/geolife_extracted/trjs_train.npy --labels_path ./data/geolife_extracted/labels_train.npy --seg_size 200 --data_type train --save_dir ./data/geolife_features
python ./trajectory_segmentation_and_features_extraction.py --trjs_path ./data/geolife_extracted/trjs_test.npy --labels_path ./data/geolife_extracted/labels_test.npy --seg_size 200 --data_type test --save_dir ./data/geolife_features
python ./MF_RP_mat_h5support.py --dim 3 --tau 8 --multi_feature_segs_path ./data/geolife_features/multi_feature_segs_train.npy --save_path ./data/geolife_features/multi_channel_RP_mats_train.h5
python ./MF_RP_mat_h5support.py --dim 3 --tau 8 --multi_feature_segs_path ./data/geolife_features/multi_feature_segs_test.npy --save_path ./data/geolife_features/multi_channel_RP_mats_test.h5
python ./PEDCC.py --save_dir ./data/geolife_features --dim 304
python network_training.py --dataset geolife --results_path ./results/test_torch_geo --alpha 1 --beta 8 --gamma 1 --RP_emb_dim 152 --FS_emb_dim 152 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0
python network_training.py --dataset geolife --results_path ${results_path} --alpha 1 --beta 8 --gamma 1 --RP_emb_dim 152 --FS_emb_dim 152 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0
