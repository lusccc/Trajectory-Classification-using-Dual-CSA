#python geolife_trajectory_extraction.py
python trajectory_segmentation_and_features.py --trjs_path "data/geolife_extracted/trjs_test.npy" --labels_path "data/geolife_extracted/labels_test.npy" --save_file_suffix "test_rdp0.8" --random_drop_percentage 0.8
python MF_RP_mat.py --trjs_segs_features_path "data/geolife_features/trjs_segs_features_test_rdp0.8.npy" --save_path "data/geolife_features/RP_mats_test_rdp0.8.npy"
python PEDCC.py --produce_centroids_for_specific_test "data/geolife_features/trjs_segs_features_test_rdp0.8.npy" --save_path "data/geolife_features/centroids_test_rdp0.8.npy"
#python dataset_generation.py
#python Dual_SAE.py
