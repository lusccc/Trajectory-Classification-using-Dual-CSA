import argparse
import os
import pathlib

from params import *


def log(info):
    with open(os.path.join(results_path, 'log.txt'), 'a') as f:
        print('★ ', end='')
        print(info)
        print('★ ', end='', file=f)
        print(info, file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DSAE')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--results_path', default='./results/default', type=str)
    parser.add_argument('--alpha', default=ALPHA, type=float)
    parser.add_argument('--beta', default=BETA, type=float)
    parser.add_argument('--gamma', default=GAMMA, type=float)
    parser.add_argument('--no_pre', default=False, type=bool)
    parser.add_argument('--no_joint', default=False, type=bool)
    parser.add_argument('--epoch1', default=3000, type=int)
    parser.add_argument('--epoch2', default=3000, type=int)
    parser.add_argument('--RP_emb_dim', type=int)
    parser.add_argument('--ts_emb_dim', type=int)
    parser.add_argument('--n_trainset_split_parts', type=int, default=1)

    args = parser.parse_args()
    results_path = args.results_path
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    no_pretrain = args.no_pre
    no_joint_train = args.no_joint
    epoch1 = args.epoch1
    epoch2 = args.epoch2
    RP_emb_dim = args.RP_emb_dim
    ts_emb_dim = args.ts_emb_dim
    TOTAL_EMB_DIM = RP_emb_dim + ts_emb_dim
    dataset = args.dataset
    n_trainset_split_parts = args.n_trainset_split_parts

    pathlib.Path(os.path.join(results_path, 'visualization')).mkdir(parents=True, exist_ok=True)

    log(f'dataset:{dataset}, results_path:{results_path} , loss weight:{alpha},{beta},{gamma},'
        f'RP_emb_dim:{RP_emb_dim}, ts_emb_dim:{ts_emb_dim}, no_pretrain:{no_pretrain}, no_joint_train:{no_pretrain}, '
        f'n_trainset_split_parts:{n_trainset_split_parts}')