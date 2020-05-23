#python trajectory_extraction.py
python trajectory_segmentation_and_features.py --trjs_path "data/geolife_extracted/trjs_test.npy" --labels_path "data/geolife_extracted/labels_test.npy" --save_file_suffix test
python MF_RP_mat.py --trjs_segs_features_path "data/geolife_features/trjs_segs_features_test.npy" --save_path "data/geolife_features/RP_mats_test.npy"
python PEDCC.py
#python dataset_generation.py
#python Dual_SAE.py
