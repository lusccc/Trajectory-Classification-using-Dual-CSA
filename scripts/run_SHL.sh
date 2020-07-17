python trajectory_segmentation_and_features.py --trjs_path "data/SHL_extracted/trjs_train.npy" --labels_path "data/SHL_extracted/labels_train.npy"  --save_file_suffix train --dataset SHL
python trajectory_segmentation_and_features.py --trjs_path "data/SHL_extracted/trjs_test.npy" --labels_path "data/SHL_extracted/labels_test.npy" --save_file_suffix test --dataset SHL
python MF_RP_mat.py --trjs_segs_features_path "data/SHL_features/trjs_segs_features_train.npy" --save_path "data/SHL_features/RP_mats_train.npy"
python MF_RP_mat.py --trjs_segs_features_path "data/SHL_features/trjs_segs_features_test.npy" --save_path "data/SHL_features/RP_mats_test.npy"
python PEDCC.py --dataset SHL
python Dual_SAE.py  --dataset SHL