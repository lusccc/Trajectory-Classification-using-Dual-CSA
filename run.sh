rm -rf ./logs
rm -rf ./results/*
python trajectory_features_and_segmentation.py
python RP_mat.py
python PEDCC.py
python Dual_SAE.py
#python Dual_SAE_LSTM.py