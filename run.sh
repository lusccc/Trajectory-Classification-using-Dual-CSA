rm -rf ./logs
rm -rf ./results/*
#python ./trajectory_segmentation_and_features.py
#python MF_RP_mat.py
python PEDCC.py
python Dual_SAE.py
#python Dual_SAE_LSTM.py