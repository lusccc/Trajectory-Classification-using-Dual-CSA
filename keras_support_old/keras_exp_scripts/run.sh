cd ..
#python trajectory_extraction.py
python ./trajectory_segmentation_and_features.py --trjs_path ./data/geolife_extracted/trjs_train.npy --labels_path ./data/geolife_extracted/labels_train.npy --data_type train --save_dir ./data/geolife_features
python ./trajectory_segmentation_and_features.py --trjs_path ./data/geolife_extracted/trjs_test.npy --labels_path ./data/geolife_extracted/labels_test.npy --data_type test --save_dir ./data/geolife_features
python ./MF_RP_mat.py --trjs_segs_features_path ./data/geolife_features/trjs_segs_features_train.npy --save_path ./data/geolife_features/RP_mats_train.npy
python ./MF_RP_mat.py --trjs_segs_features_path ./data/geolife_features/trjs_segs_features_test.npy --save_path ./data/geolife_features/RP_mats_test.npy
python ./PEDCC.py --features_path ./data/geolife_features/trjs_segs_features_train.npy --save_dir ./data/geolife_features --data_type train --dim 96
python ./PEDCC.py --features_path ./data/geolife_features/trjs_segs_features_test.npy --save_dir ./data/geolife_features --data_type test --dim 96
python -m network.Dual_SAE --dataset geolife
