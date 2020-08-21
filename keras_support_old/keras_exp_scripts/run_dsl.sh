rm -rf ./logs
mkdir ./results_dsl
rm -rf ./results_dsl/*
mkdir ./results_dsl/visualization
#python ./trajectory_segmentation_and_features_extraction.py
#python MF_RP_mat.py
#python PEDCC.py
#python dataset_generation.py
python Dual_SAE_LSTM.py
