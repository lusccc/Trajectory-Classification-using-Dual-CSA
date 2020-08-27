import argparse
import importlib
import os
import pathlib
import timeit
from math import inf

import logzero
import numpy as np
from logzero import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torchsummary import summary
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from logzero import logger

pathlib.Path(os.environ['RES_PATH']).mkdir(parents=True, exist_ok=True)
logzero.logfile(os.path.join(os.environ['RES_PATH'], 'log.txt'), backupCount=3)


def cleanup(args):
    logger.info(get_log_str(args, 'cleaning up...'))
    dist.destroy_process_group()


def get_log_str(args, str):
    return f'[node:{args.node} rank:{args.rank}] {str}'


def try_to_save_model(args, model, path):
    # this condition statement means only save model on first gpu (rank=0),
    # to avoid save multiple time and cause parallel writing/reading issue
    if not args.no_save_model and (not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                                            and args.rank % args.ngpus_per_node == 0)):
        logger.info(get_log_str(args, f'saving model using node {args.node} rank {args.rank}, path: {path}'))
        torch.save(model.state_dict(), path)

    # Use a barrier() to make sure that other process loads the model after process
    # 0 saves it.
    # https://stackoverflow.com/questions/59760328/how-does-torch-distributed-barrier-work
    dist.barrier()


def load_saved_model(args, path, model):
    logger.info(get_log_str(args, f'loading saved model on gpu: {args.gpu}, path: {path}'))
    if args.gpu is None:
        model.load_state_dict(torch.load(path))
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        model.load_state_dict(torch.load(path, map_location=loc))
    return model


class Trajectory_Feature_Segment_Dataset(Dataset):
    def __init__(self, dataset_name, data_type):
        self.dataset_name = dataset_name
        self.data_type = data_type
        self.multi_feature_segs = torch.from_numpy(
            np.load(f'./data/{self.dataset_name}_features/multi_feature_segs_{data_type}_normalized.npy')).float()
        self.labels = torch.from_numpy(
            np.load(f'./data/{self.dataset_name}_features/multi_feature_seg_labels_{data_type}.npy')).float()

    def __len__(self):
        return self.multi_feature_segs.shape[0]

    def __getitem__(self, index):
        label = self.labels[index]
        FS = self.multi_feature_segs[index]
        return FS, label


def get_network(name):
    network = getattr(importlib.import_module("compared_network"), name)
    return network()

def train(args, model, train_loader, test_loader, train_sampler=None, device=None):
    logger.info(get_log_str(args, 'training...'))
    train_start = timeit.time.perf_counter()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    best_score = inf
    not_improving_step = 0

    for epoch in range(args.start_epoch, args.epochs, ):
        epoch_start = timeit.time.perf_counter()

        accum_train_loss = 0
        accum_test_loss = 0

        # in DistributedDataParallel training, the size of train_sampler.dataset is not equal to actual sample number
        n_total_train_samples = 0
        n_total_test_samples = 0

        # train model
        model.train()
        train_sampler.set_epoch(epoch)
        for batch_idx, (FS, true_label) in enumerate(train_loader):
            batch_start = timeit.time.perf_counter()
            FS, true_label = FS.to(device), true_label.to(device)
            pred_label = model(FS)
            optimizer.zero_grad()
            loss =





def main_worker(proc, args):
    args.gpu = proc
    args.rank = proc  # this experiment only designed for single node, hence rank set to gpu id
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.gpu)

    model = get_network(args.network)
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    device = f'cuda:{args.gpu}'
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    train_dataset = Trajectory_Feature_Segment_Dataset(args.dataset, 'train', )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, sampler=train_sampler, pin_memory=True)
    test_loader = DataLoader(Trajectory_Feature_Segment_Dataset(args.dataset, 'test', ),
                             batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers, pin_memory=True)

    cudnn.benchmark = True
    if args.evaluate:
        pass
    if os.path.exists(os.path.join(args.results_path, f'{args.network}.pt')):
        logger.info(get_log_str(args, f'{args.network}.pt exists, continue training from last time!'))
        model = load_saved_model(args, os.path.join(args.results_path, f'{args.network}.pt'), model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='network comparison')
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on testation set')

    args = parser.parse_args()
    args.dist_backend = 'nccl'
    args.dist_url = 'tcp://127.0.0.1:6666'
    args.world_size = 2
    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
