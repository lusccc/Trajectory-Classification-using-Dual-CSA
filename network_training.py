import argparse
import os
import pathlib
import random
import sys
from math import inf

import torch.distributed as dist
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import dataset_factory
from netwok_torch.Dual_CSA import Dual_CSA
from params import *
import numpy as np
import logzero
from logzero import logger

logzero.logfile(os.path.join(os.environ['RES_PATH'], 'log.txt'), maxBytes=1e6, backupCount=3)

'''
Some of the code about DistributedDataParallel is referenced from:
https://github.com/pytorch/examples/tree/master/imagenet
'''


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pretrain(args, model, train_loader, test_loader, train_sampler=None, device=None):
    logger.info('pretraining...')
    model.pretrain = True
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    recon_criterion = nn.MSELoss()

    best_score = inf
    not_improving_step = 0

    for epoch in range(args.start_epoch, args.epochs, ):
        accum_train_loss = 0
        accum_test_loss = 0

        # in DistributedDataParallel training, the size of train_sampler.dataset is not equal to actual sample number
        n_total_train_samples = 0
        n_total_test_samples = 0

        # train model
        model.train()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        for batch_idx, (RP, FS, _) in enumerate(train_loader):
            RP, FS = RP.to(device), FS.to(device)
            optimizer.zero_grad()
            RP_recon, __, FS_recon = model(RP, FS)
            loss = recon_criterion(RP_recon, RP) + recon_criterion(FS_recon, FS)
            loss.backward()
            optimizer.step()
            n_total_train_samples += len(RP)
            accum_train_loss += loss.item() * len(RP)
            align_width = len(str(len(train_loader)))
            if batch_idx % args.print_freq == 0:
                logger.info(f'[device:{device}] train Epoch: {epoch}, '
                            f'[{batch_idx:>{align_width}}'
                            f'/{len(train_loader):>{align_width}}], '
                            f'\tLoss: {loss.item():.6f}')

        # eval model
        model.eval()
        with torch.no_grad():
            for RP, FS, _ in test_loader:
                RP, FS = RP.to(device), FS.to(device)
                RP_recon, __, FS_recon = model(RP, FS)
                loss = recon_criterion(RP_recon, RP) + recon_criterion(FS_recon, FS)
                n_total_test_samples += len(RP)
                accum_test_loss += loss.item() * len(RP)

        train_loss = accum_train_loss / n_total_train_samples
        test_loss = accum_test_loss / n_total_test_samples
        logger.info(f'[device:{device}] ==> ' +
                    f'avg train_loss: {train_loss:.5f}, ' +
                    f'avg test_loss: {test_loss:.5f}')

        # early stop
        if test_loss < best_score:
            not_improving_step = 0
            logger.info(f'[device:{device}] * score improved from {best_score:.5f} to {test_loss:.5f}, saving model...')
            torch.save(model.state_dict(), os.path.join(args.results_path, 'pretrained_AE.pt'))
            best_score = test_loss
        else:
            not_improving_step += 1
            if not_improving_step > args.patience:
                logger.info(f'[device:{device}] * best score:{best_score}, not_improving_step:{not_improving_step}, early stop!')
            else:
                logger.info(f'[device:{device}] * best:{best_score}, not_improving_step:{not_improving_step}')


def joint_train(args, model, train_loader, test_loader, train_sampler=None, loss_weight=[1, 1, 1], device=None):
    logger.info('joint training...')
    model.module.pretrain = False
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)  # weight_decay added
    recon_criterion = nn.MSELoss()
    pedcc_criterion = nn.KLDivLoss(reduction='batchmean')
    alpha, beta, gamma = loss_weight

    best_score = 0
    not_improving_step = 0

    for epoch in range(args.start_epoch, args.epochs, ):
        n_train_pred_correct = 0
        n_test_pred_correct = 0

        accum_train_loss = 0
        accum_test_loss = 0

        # in DistributedDataParallel training, the size of train_sampler.dataset is not equal to actual sample number
        n_total_train_samples = 0
        n_total_test_samples = 0

        # train model
        model.train()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        for batch_idx, (RP, FS, true_label) in enumerate(train_loader):
            RP, FS, true_label = RP.to(device), FS.to(device), true_label.to(device)
            optimizer.zero_grad()
            RP_recon, soft_label, FS_recon = model(RP, FS)
            RP_recon_loss = recon_criterion(RP_recon, RP)
            pedcc_classification_loss = pedcc_criterion(torch.log(soft_label).float(), true_label)
            FS_recon_loss = recon_criterion(FS_recon, FS)
            combine_loss = alpha * RP_recon_loss + beta * pedcc_classification_loss + gamma * FS_recon_loss
            combine_loss = combine_loss
            combine_loss.backward()
            optimizer.step()
            accum_train_loss += combine_loss.item() * len(RP)
            pred = soft_label.argmax(dim=1, keepdim=True)
            true_label = true_label.argmax(dim=1, keepdim=True)
            n_train_pred_correct += pred.eq(true_label).sum().item()
            n_total_train_samples += len(RP)
            align_width = len(str(len(train_loader)))
            if batch_idx % args.print_freq == 0:
                logger.info(f'[device:{device}] train Epoch: {epoch}, '
                            f'[{batch_idx:>{align_width}} '
                            f'/{len(train_loader):>{align_width}}], '
                            f'RP_recon_loss: {RP_recon_loss.item():.6f}, '
                            f'pedcc_classification_loss: {pedcc_classification_loss.item():.6f}, '
                            f'FS_recon_loss: {FS_recon_loss.item():.6f},'
                            f'combine_loss: {combine_loss.item():.6f}')

        # eval model
        model.eval()
        with torch.no_grad():
            for batch_idx, (RP, FS, true_label) in enumerate(test_loader):
                RP, FS, true_label = RP.to(device), FS.to(device), true_label.to(device)
                RP_recon, soft_label, FS_recon = model(RP, FS)
                RP_recon_loss = recon_criterion(RP_recon, RP)
                pedcc_classification_loss = pedcc_criterion(torch.log(soft_label), true_label)
                FS_recon_loss = recon_criterion(FS_recon, FS)
                combine_loss = alpha * RP_recon_loss + beta * pedcc_classification_loss + gamma * FS_recon_loss
                accum_test_loss += combine_loss.item() * len(RP)
                pred = soft_label.argmax(dim=1, keepdim=True)
                true_label = true_label.argmax(dim=1, keepdim=True)
                n_test_pred_correct += pred.eq(true_label).sum().item()
                n_total_test_samples += len(RP)

        train_loss = accum_train_loss / n_total_train_samples
        test_loss = accum_test_loss / n_total_test_samples
        train_acc = n_train_pred_correct / n_total_train_samples
        test_acc = n_test_pred_correct / n_total_test_samples
        logger.info(f'[device:{device}] ==> ' +
                    f'avg train_loss: {train_loss:.5f}, '
                    f'train acc: {train_acc:.5f}, '
                    f'avg test_loss: {test_loss:.5f}, '
                    f'test_acc: {test_acc:.5f}')

        # early stop
        if test_acc > best_score:
            not_improving_step = 0
            logger.info(f'[device:{device}] * score improved from {best_score:.5f} to {test_acc:.5f}, saving model...')
            torch.save(model.state_dict(), os.path.join(args.results_path, 'Dual_CSA.pt'))
            best_score = test_acc
        else:
            not_improving_step += 1
            if not_improving_step > args.patience:
                logger.info(
                    f'[device:{device}] * best score:{best_score}, not_improving_step:{not_improving_step}, early stop!')
            else:
                logger.info(f'[device:{device}] * best:{best_score}, not_improving_step:{not_improving_step}')


def test_predict(args, model, test_loader, loss_weight=[1, 1, 1], device=None):
    logger.info('test_predict...')
    model.eval()
    model.module.pretrain = False
    total_n_corrects = 0
    with torch.no_grad():
        alpha, beta, gamma = loss_weight
        recon_criterion = nn.MSELoss()
        pedcc_criterion = nn.KLDivLoss(reduction='batchmean')
        test_losses = []
        pred_labels = []
        true_labels = []
        for batch_idx, (RP, FS, true_label) in enumerate(test_loader):
            RP, FS, true_label = RP.to(device), FS.to(device), true_label.to(device)
            RP_recon, soft_label, FS_recon = model(RP, FS)
            RP_recon_loss = recon_criterion(RP_recon, RP)
            pedcc_classification_loss = pedcc_criterion(torch.log(soft_label), true_label)
            FS_recon_loss = recon_criterion(FS_recon, FS)
            total_loss = alpha * RP_recon_loss + beta * pedcc_classification_loss + gamma * FS_recon_loss
            test_losses.append(total_loss.item())
            pred = soft_label.argmax(dim=1, keepdim=True)
            pred_labels.extend(pred.tolist())
            true_label = true_label.argmax(dim=1, keepdim=True)
            true_labels.extend(true_label.tolist())
            n_corrects = pred.eq(true_label).sum().item()
            total_n_corrects += n_corrects

        test_loss = np.average(test_losses)
        test_acc = total_n_corrects / len(test_loader.dataset)
        logger.info(f'[device:{device}] ==> ' +
                    f'avg test_loss: {test_loss:.5f}, '
                    f'test_acc: {test_acc:.5f}')
        logger.info(f'?!{total_n_corrects / len(test_loader.dataset)}')
        show_classification_results(args, pred_labels, true_labels)


def show_classification_results(args, y_pred, y_true, label_names=['walk', 'bike', 'bus', 'driving', 'train/subway']):
    logger.info('show_classification_results:')
    cm = confusion_matrix(y_true, y_pred, labels=modes_to_use)
    re = classification_report(y_true, y_pred, target_names=['walk', 'bike', 'bus', 'driving', 'train/subway'],
                               digits=5)
    print('confusion_matrix:\r' + str(re))
    print('classification_report:\r' + str(cm))
    with open(os.path.join(args.results_path, 'classification_results.txt'), 'w') as f:
        print(cm, file=f)
        print(re, file=f)


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
    device = 'cpu'
    centroid = torch.from_numpy(np.load(f'./data/{args.dataset}_features/pedcc.npy'))
    model = Dual_CSA(args.n_channels, args.RP_emb_dim, args.FS_emb_dim, centroid)
    if not torch.cuda.is_available():
        logger.warning('using CPU, this will be slow')
        device = 'cpu'
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            device = f'cuda:{args.gpu}'
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
            device = f'cuda'

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        device = f'cuda:{args.gpu}'
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model.centroid.cuda()
        model = torch.nn.DataParallel(model).cuda()
        device = f'cuda'
    logger.info(f'device: {device}')

    # loading data
    train_dataset = dataset_factory.Trajectory_Feature_Dataset(args.dataset, 'train', )
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, sampler=train_sampler, pin_memory=True)
    test_loader = DataLoader(dataset_factory.Trajectory_Feature_Dataset(args.dataset, 'test', ),
                             batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers, pin_memory=True)

    cudnn.benchmark = True
    if args.evaluate:
        model.load_state_dict(torch.load(os.path.join(args.results_path, 'Dual_CSA.pt')))
        test_predict(args, model, test_loader, [args.alpha, args.beta, args.gamma], device)
        return

    if args.pretrained:
        # default is load Dual_CSA.pt to continue joint training Dual_CSA
        if os.path.exists(os.path.join(args.results_path, 'Dual_CSA.pt')):
            logger.info(f'Dual_CSA.pt exists, continue joint training from last time!')
            model.load_state_dict(torch.load(os.path.join(args.results_path, 'Dual_CSA.pt')))
        else:
            model.load_state_dict(torch.load(os.path.join(args.results_path, 'pretrained_AE.pt')))
        joint_train(args, model, train_loader, test_loader, train_sampler, [args.alpha, args.beta, args.gamma], device)
    else:
        if os.path.exists(os.path.join(args.results_path, 'pretrained_AE.pt')):
            logger.info(f'pretrained_AE.pt exists, continue pretraining from last time!')
            model.load_state_dict(torch.load(os.path.join(args.results_path, 'pretrained_AE.pt')))
        pretrain(args, model, train_loader, test_loader, train_sampler, device)
        # load pretrained model
        model.load_state_dict(torch.load(os.path.join(args.results_path, 'pretrained_AE.pt')))
        joint_train(args, model, train_loader, test_loader, train_sampler, [args.alpha, args.beta, args.gamma], device)

    # pretrain(args, model, train_loader, test_loader, train_sampler, device)
    # model.load_state_dict(torch.load(os.path.join(args.results_path, 'pretrained_AE.pt')))
    # model.load_state_dict(torch.load(os.path.join(args.results_path, 'Dual_CSA.pt')))


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
    parser.add_argument('--RP_emb_dim', type=int)
    parser.add_argument('--FS_emb_dim', type=int)
    parser.add_argument('--patience', type=int, default=15)

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
                        help='evaluate model on testation set')
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
    os.environ['RES_PATH'] = args.results_path
    pathlib.Path(os.path.join(args.results_path, 'visualization')).mkdir(parents=True, exist_ok=True)
    logger.info('params: ' + str(args))

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
