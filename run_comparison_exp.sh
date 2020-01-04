rm -rf ./logs
rm -rf ./comparison_results/*
python ./trajectory_segmentation_and_features.py
python ./MF_RP_mat.py
python ./PEDCC.py
python Dual_Softmax_AE.py