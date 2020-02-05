rm -rf ./logs
mkdir ./results
rm -rf ./results/*
mkdir ./results/visualization
#python trajectory_extraction.py
#python trajectory_segmentation_and_features.py
python MF_RP_mat.py
python PEDCC.py
python dataset.py
python Dual_SAE.py