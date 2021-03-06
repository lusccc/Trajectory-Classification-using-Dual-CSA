cd ..
python ./trajectory_segmentation_and_features.py --trjs_path ./data/SHL_extracted/trjs_train.npy --labels_path ./data/SHL_extracted/labels_train.npy --seg_size 152 --data_type train --save_dir ./data/SHL_features
python ./trajectory_segmentation_and_features.py --trjs_path ./data/SHL_extracted/trjs_test.npy --labels_path ./data/SHL_extracted/labels_test.npy --seg_size 152 --data_type test --save_dir ./data/SHL_features
python ./MF_RP_mat.py --dim 3 --tau 7 --trjs_segs_features_path ./data/SHL_features/trjs_segs_features_train.npy --save_path ./data/SHL_features/RP_mats_train.npy
python ./MF_RP_mat.py --dim 3 --tau 7 --trjs_segs_features_path ./data/SHL_features/trjs_segs_features_test.npy --save_path ./data/SHL_features/RP_mats_test.npy
python ./PEDCC.py --features_path ./data/SHL_features/trjs_segs_features_train.npy --save_dir ./data/SHL_features --data_type train --dim 304
python ./PEDCC.py --features_path ./data/SHL_features/trjs_segs_features_test.npy --save_dir ./data/SHL_features --data_type test --dim 304
python -m network.Dual_SAE --dataset SHL --results_path './results/exp_SHL_staypoint_split_152_3,7_304_1,8,1' --alpha 1 --beta 8 --gamma 1 --RP_emb_dim 152 --ts_emb_dim 152
