rm -rf ./logs
mkdir ./results_dsn
rm -rf ./results_dsn/*
mkdir ./results_dsn/visualization
#python ./trajectory_segmentation_and_features.py
#python MF_RP_mat.py
#python PEDCC.py
#python dataset.py
python Dual_SAE_no_pre_or_joint_train.py