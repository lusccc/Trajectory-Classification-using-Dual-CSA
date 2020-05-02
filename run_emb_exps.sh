for i in `seq 16 8 192`
do
    mkdir -p "./results/EMB${i}/"
    cp ./results/RP_conv_ae_check_point.model "./results/EMB${i}/"
    cp ./results/ts_conv_ae_check_point.model "./results/EMB${i}/"
    python dataset_generation.py --DIM ${i}
    python Dual_SAE.py --results_path "./results/EMB${i}/"
done
