rm -rf ./logs
mkdir ./results_dsn
rm -rf ./results_dsn/*
mkdir ./results_dsn/visualization
#python ./trajectory_segmentation_and_features_extraction.py
#python MF_RP_mat.py
#python PEDCC.py
#python dataset_generation.py
python Dual_SAE_no_pre_or_joint_train.py
