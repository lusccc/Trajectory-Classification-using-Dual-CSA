
# loss weight:1,3,1

cd ..
for i in `seq 96 8 128`
do
    path="./results/exp1st_SHL_t7s3e${i}/"
    mkdir -p ${path}
    # copy trained autoencoder weight files. no need to pretrain
    cp ./results/best_dual_ae_model/RP_conv_ae_check_point.model ${path}
    cp ./results/best_dual_ae_model/ts_conv_ae_check_point.model ${path}
    python ./PEDCC.py --features_path ./data/SHL_features/trjs_segs_features_train.npy  --save_dir ./data/SHL_features --data_type train --dim ${i}
    python ./PEDCC.py --features_path ./data/SHL_features/trjs_segs_features_test.npy  --save_dir ./data/SHL_features --data_type test --dim ${i}

    python -m network.Dual_SAE --dataset SHL --results_path ${path}
done

for i in `seq 8 8 128`
do
    path="./results/exp2nd_SHL_t7s3e${i}/"
    mkdir -p ${path}
    # copy trained autoencoder weight files. no need to pretrain
    cp ./results/best_dual_ae_model/RP_conv_ae_check_point.model ${path}
    cp ./results/best_dual_ae_model/ts_conv_ae_check_point.model ${path}
    python ./PEDCC.py --features_path ./data/SHL_features/trjs_segs_features_train.npy  --save_dir ./data/SHL_features --data_type train --dim ${i}
    python ./PEDCC.py --features_path ./data/SHL_features/trjs_segs_features_test.npy  --save_dir ./data/SHL_features --data_type test --dim ${i}

    python -m network.Dual_SAE --dataset SHL --results_path ${path}
done


for i in `seq 8 8 128`
do
    path="./results/exp3rd_SHL_t7s3e${i}/"
    mkdir -p ${path}
    # copy trained autoencoder weight files. no need to pretrain
    cp ./results/best_dual_ae_model/RP_conv_ae_check_point.model ${path}
    cp ./results/best_dual_ae_model/ts_conv_ae_check_point.model ${path}
    python ./PEDCC.py --features_path ./data/SHL_features/trjs_segs_features_train.npy  --save_dir ./data/SHL_features --data_type train --dim ${i}
    python ./PEDCC.py --features_path ./data/SHL_features/trjs_segs_features_test.npy  --save_dir ./data/SHL_features --data_type test --dim ${i}

    python -m network.Dual_SAE --dataset SHL --results_path ${path}
done

for i in `seq 8 8 128`
do
    path="./results/exp4th_SHL_t7s3e${i}/"
    mkdir -p ${path}
    # copy trained autoencoder weight files. no need to pretrain
    cp ./results/best_dual_ae_model/RP_conv_ae_check_point.model ${path}
    cp ./results/best_dual_ae_model/ts_conv_ae_check_point.model ${path}
    python ./PEDCC.py --features_path ./data/SHL_features/trjs_segs_features_train.npy  --save_dir ./data/SHL_features --data_type train --dim ${i}
    python ./PEDCC.py --features_path ./data/SHL_features/trjs_segs_features_test.npy  --save_dir ./data/SHL_features --data_type test --dim ${i}

    python -m network.Dual_SAE --dataset SHL --results_path ${path}
done
