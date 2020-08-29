# dim=3 tau=8;emb=304;beta=11
cd ..
path="./results/dca_softmax_geolife/"
mkdir -p ${path}
export RES_PATH=${path}
mkdir -p ${path}
#cp ./results/geolife_pretrained_emb304/pretrained_AE.pt ${path}
#python ./trajectory_segmentation_and_features_extraction.py --trjs_path ./data/geolife_extracted/trjs_train.npy --labels_path ./data/geolife_extracted/labels_train.npy --seg_size 200 --data_type train --save_dir ./data/geolife_features
#python ./trajectory_segmentation_and_features_extraction.py --trjs_path ./data/geolife_extracted/trjs_test.npy --labels_path ./data/geolife_extracted/labels_test.npy --seg_size 200 --data_type test --save_dir ./data/geolife_features
#python ./MF_RP_mat_h5support.py --dim 3 --tau 8 --multi_feature_segs_path ./data/geolife_features/multi_feature_segs_train.npy --save_path ./data/geolife_features/multi_channel_RP_mats_train.h5
#python ./MF_RP_mat_h5support.py --dim 3 --tau 8 --multi_feature_segs_path ./data/geolife_features/multi_feature_segs_test.npy --save_path ./data/geolife_features/multi_channel_RP_mats_test.h5
python ./PEDCC.py --save_dir ./data/geolife_features --dim 304
python network_training.py --dataset geolife --results-path ${path} --alpha 1 --beta 11 --gamma 1 --RP-emb-dim 152 --FS-emb-dim 152 --patience 20 --dist-url tcp://127.0.0.1:6667 --dist-backend nccl --gpu 1 --world-size 1 --rank 0 -b 160 --network Dual_CA_Softmax
