cd ..
#python ./trajectory_segmentation_and_features.py --trjs_path ./data/SHL_extracted/trjs_train.npy --labels_path ./data/SHL_extracted/labels_train.npy --seg_size 152 --data_type train --save_dir ./data/SHL_features
#python ./trajectory_segmentation_and_features.py --trjs_path ./data/SHL_extracted/trjs_test.npy --labels_path ./data/SHL_extracted/labels_test.npy --seg_size 152 --data_type test --save_dir ./data/SHL_features
python ./MF_RP_mat.py --dim 3 --tau 7 --trjs_segs_features_path ./data/SHL_features/trjs_segs_features_train.npy --save_path ./data/SHL_features/RP_mats_train.npy
python ./MF_RP_mat.py --dim 3 --tau 7 --trjs_segs_features_path ./data/SHL_features/trjs_segs_features_test.npy --save_path ./data/SHL_features/RP_mats_test.npy
python ./PEDCC.py --features_path ./data/SHL_features/trjs_segs_features_train.npy --save_dir ./data/SHL_features --data_type train --dim 96
python ./PEDCC.py --features_path ./data/SHL_features/trjs_segs_features_test.npy --save_dir ./data/SHL_features --data_type test --dim 96
python -m network.Dual_SAE --dataset SHL
