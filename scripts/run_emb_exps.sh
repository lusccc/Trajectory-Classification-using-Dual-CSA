# other parameters fixed to:
#dim=tau=9, loss_weight=1,3,1
cd ..
for i in `seq 8 8 192`
do
    path="./results/rep1_emb${i}/"
    mkdir -p ${path}
    # copy trained autoencoder weight files. no need to pretrain
    cp ./results/best_dual_ae_model/RP_conv_ae_check_point.model ${path}
    cp ./results/best_dual_ae_model/ts_conv_ae_check_point.model ${path}
    python ./PEDCC.py --features_path ./data/geolife_features/trjs_segs_features_train.npy  --save_dir ./data/geolife_features --data_type train --dim ${i}
    python ./PEDCC.py --features_path ./data/geolife_features/trjs_segs_features_test.npy  --save_dir ./data/geolife_features --data_type test --dim ${i}

    python -m network.Dual_SAE --dataset geolife --results_path ${path}
done
