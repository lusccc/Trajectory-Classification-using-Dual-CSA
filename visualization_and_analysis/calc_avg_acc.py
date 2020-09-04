import os
import re

import matplotlib.pyplot as plt
import numpy as np

res_paths = [
    'D:/exps_results_useful/geolife_rnn_softmax_keras_exps',
    'D:/exps_results_useful/geolife_only_RP_AE_exps',
    'D:/exps_results_useful/geolife_only_FS_AE_exps',
    'D:/exps_results_useful/geolife_only_joint_exps',
    'D:/exps_results_useful/geolife_lstm_softmax_keras_exps',
    'D:/exps_results_useful/geolife_lstm_fcn_softmax_keras_exps',
    'D:/exps_results_useful/geolife_optimal_exps',
    'D:/exps_results_useful/geolife_no_pre_joint_train_exps',
    'D:/exps_results_useful/geolife_drop_decoder_exps',
    'D:/exps_results_useful/geolife_manual_select_pretrain_epochs5_dynamic_loss_weight_drop_to_0,1,0_exps',
    'D:/exps_results_useful/geolife_manual_select_pretrain_epochs5_fixed_loss_weight_1,1,1_exps',
    'D:/exps_results_useful/geolife_manual_select_pretrain_epochs5_dynamic_loss_weight_1,1,1_to_0,1,0_exps',
    'D:/exps_results_useful/SHL_lstm_fcn_softmax_keras_exps',
    'D:/exps_results_useful/SHL_lstm_softmax_keras_exps',
    'D:/exps_results_useful/SHL_only_FS_AE_exps',
    'D:/exps_results_useful/SHL_only_joint_train_exps',
    'D:/exps_results_useful/SHL_only_RP_AE_exps',
    'D:/exps_results_useful/SHL_optimal_exps',
    'D:/exps_results_useful/SHL_rnn_softmax_keras_exps',
    'D:/exps_results_useful/SHL_no_pre_joint_train_exps',
    'D:/exps_results_useful/SHL_drop_decoder_exps',
    'D:/exps_results_useful/SHL_manual_select_pretrain_epochs100_exps',
    'D:/exps_results_useful/SHL_manual_select_pretrain_epochs100_dynamic_loss_weight_drop_to_0,1,0_exps',
    'D:/exps_results_useful/SHL_manual_select_pretrain_epochs100_dynamic_loss_weight_with_reset_exps',
    'D:/exps_results_useful/SHL_manual_select_pretrain_epochs100_fixed_loss_weight_1,1,1_exps',
    'D:/exps_results_useful/SHL_manual_select_pretrain_epochs100_dynamic_loss_weight_1,1,1_to_0,1,0_exps',
]
res_subpaths = [
    'geolife_rnn_softmax_exp',
    'geolife_only_RP_AE_exp',
    'geolife_only_FS_AE_exp',
    'geolife_only_joint_exp',
    'geolife_lstm_softmax_exp',
    'geolife_lstm_fcn_softmax_exp',
    'geolife_optimal_exp',
    'geolife_no_pre_joint_exp',
    'geolife_drop_decoder_exp',
    'geolife_manual_select_pretrain_epochs5_dynamic_loss_weight_drop_to_0,1,0_exp',
    'geolife_manual_select_pretrain_epochs5_fixed_loss_weight_1,1,1_exp',
    'geolife_manual_select_pretrain_epochs5_dynamic_loss_weight_1,1,1_to_0,1,0_exp',
    'SHL_lstm_fcn_softmax_exp',
    'SHL_lstm_softmax_exp',
    'SHL_only_FS_AE_exp',
    'SHL_only_joint_exp',
    'SHL_only_RP_AE_exp',
    'SHL_optimal_exp',
    'SHL_rnn_softmax_exp',
    'SHL_no_pre_joint_exp',
    'SHL_drop_decoder_exp',
    'SHL_manual_select_pretrain_epochs100_exp',
    'SHL_manual_select_pretrain_epochs100_dynamic_loss_weight_drop_to_0,1,0_exp',
    'SHL_manual_select_pretrain_epochs100_dynamic_loss_weight_with_reset_exp',
    'SHL_manual_select_pretrain_epochs100_fixed_loss_weight_1,1,1_exp',
    'SHL_manual_select_pretrain_epochs100_dynamic_loss_weight_1,1,1_to_0,1,0_exp',
]
for j, res_path in enumerate(res_paths):
    path, dirs, files = next(os.walk(res_path))
    dir_count = len(dirs)
    total_accs = []
    total_f1s = []
    total_weighted_avgs = []
    for i in range(dir_count):
        exp_path = os.path.join(res_path, f'{res_subpaths[j]}{i}')
        log_path = os.path.join(exp_path, 'classification_results.txt')
        lines = open(log_path, encoding='utf-8').readlines()
        acc = float(lines[-4][36:43])  # Penultimate line, and only cut acc value
        total_accs.append(acc)
        total_f1s.append(
            # walk              bike               bus                driving         train/subway
            [lines[-10][36:43], lines[-9][36:43], lines[-8][36:43], lines[-7][36:43], lines[-6][36:43]]
        )
        total_weighted_avgs.append(float(lines[-2][36:43]))
    mean_acc = np.mean(total_accs, axis=0)
    total_f1s = np.array(total_f1s, dtype='float')
    np.set_printoptions(formatter={'float_kind': "{:.5f}".format})
    mean_f1 = np.mean(total_f1s, axis=0)
    mean_weighted_avg = np.mean(total_weighted_avgs, axis=0)
    print('-' * 250)
    print(f'{res_subpaths[j]:<90} mean_acc: {mean_acc:<10.5f} mean_f1: {str(mean_f1):<45} f1_mean_weighted_avg: {mean_weighted_avg:<10.5f} exp_times: {dir_count}')
