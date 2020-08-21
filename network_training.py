import argparse
import os
import pathlib
import random

import torch.distributed as dist
import torch
import torch.nn as nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import dataset_factory
from netwok_torch.Dual_CSA import Dual_CSA
from netwok_torch.pytorchtools import EarlyStopping
from params import *
import numpy as np
import logzero
from logzero import logger

'''
Some of the code about DistributedDataParallel is referenced from:
https://github.com/pytorch/examples/tree/master/imagenet
'''


def combine_loss_function(RP, RP_recon, soft_label, true_label, FS, FS_recon, loss_weight=[1, 1, 1]):
    recon_criterion = nn.MSELoss()
    classification_criterion = nn.KLDivLoss()
    RP_mse = recon_criterion(RP_recon, RP)
    classification_kld = classification_criterion(torch.log(soft_label), true_label)
    FS_mse = recon_criterion(FS_recon, FS)
    alpha, beta, gamma = loss_weight
    total_loss = alpha * RP_mse + beta * classification_kld + gamma * FS_mse
    return total_loss


def pretrain(args, model, train_loader, test_loader, train_sampler=None):
    model.pretrain = True
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    recon_criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=20, verbose=True, path=os.path.join(args.results_path, 'pretrained_AE.pt'),
                                   trace_func=logger.info)
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    for epoch in range(args.start_epoch, args.epochs, ):
        # train model
        print()
        model.train()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_loss = 0
        for batch_idx, (RP, FS, _) in enumerate(train_loader):
            optimizer.zero_grad()
            RP, FS = RP.float().cuda(), FS.float().cuda()
            RP_recon, __, FS_recon = model(RP, FS)
            loss = recon_criterion(RP_recon, RP) + recon_criterion(FS_recon, FS)
            loss.backward()
            train_loss += loss.item()
            train_losses.append(loss.item())
            optimizer.step()
            if batch_idx % args.print_freq == 0:
                logger.info(f'Train Epoch: {epoch}, '
                            f'[{batch_idx * len(RP):>{len(str(args.epochs))}}'
                            f'/{len(train_loader.dataset):>{len(str(args.epochs))}} '
                            f'({100. * batch_idx / len(train_loader):2.0f}%)]'
                            f'\tLoss: {loss.item():.6f}')

        # valid model
        model.eval()
        for RP, FS, _ in test_loader:
            RP, FS = RP.float().cuda(), FS.float().cuda()
            RP_recon, __, FS_recon = model(RP, FS)
            loss = recon_criterion(RP_recon, RP) + recon_criterion(FS_recon, FS)
            valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        logger.info(f'==> ' +
                    f'avg train_loss: {train_loss:.5f}, ' +
                    f'avg valid_loss: {valid_loss:.5f}')

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break


def main_worker(gpu, ngpus_per_node, args):
    # init distributed data parallel
    args.gpu = gpu
    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    centroid = torch.from_numpy(np.load(f'./data/{args.dataset}_features/pedcc.npy'))
    if args.pretrained:
        # TODO：
        pass
    else:
        model = Dual_CSA(args.n_channels, args.RP_emb_dim, args.FS_emb_dim, centroid)
    if not torch.cuda.is_available():
        logger.warning('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # loading data
    train_dataset = dataset_factory.Trajectory_Feature_Dataset(args.dataset, 'train', )
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    test_loader = DataLoader(dataset_factory.Trajectory_Feature_Dataset(args.dataset, 'test', ),
                             batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        # TODO：
        return

    pretrain(args, model, train_loader, test_loader, train_sampler)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DCSA_Training')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--results_path', default='./results/default', type=str)
    parser.add_argument('--alpha', default=ALPHA, type=float)
    parser.add_argument('--beta', default=BETA, type=float)
    parser.add_argument('--gamma', default=GAMMA, type=float)
    parser.add_argument('--n_channels', default=5, type=float)
    parser.add_argument('--no_pre', default=False, type=bool)
    parser.add_argument('--no_joint', default=False, type=bool)
    parser.add_argument('--epoch1', default=3000, type=int)
    parser.add_argument('--epoch2', default=3000, type=int)
    parser.add_argument('--RP_emb_dim', type=int)
    parser.add_argument('--FS_emb_dim', type=int)

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=3000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    args = parser.parse_args()
    pathlib.Path(os.path.join(args.results_path, 'visualization')).mkdir(parents=True, exist_ok=True)
    logzero.logfile(os.path.join(args.results_path, 'log.txt'), maxBytes=1e6, backupCount=3)
    logger.info(
        f'dataset_name:{args.dataset}, results_path:{args.results_path} , loss weight:{args.alpha},{args.beta},{args.gamma},'
        f'RP_emb_dim:{args.RP_emb_dim}, ts_emb_dim:{args.FS_emb_dim}, no_pretrain:{args.no_pre}, no_joint_train:{args.no_pre}')

    # write params to args
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        logger.warning('You have chosen to seed training. '
                       'This will turn on the CUDNN deterministic setting, '
                       'which can slow down your training considerably! '
                       'You may see unexpected behavior when restarting '
                       'from checkpoints.')

    if args.gpu is not None:
        logger.warning('You have chosen a specific GPU. This will completely '
                       'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
