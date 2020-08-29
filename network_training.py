import argparse
import os
import pathlib
import random
import timeit
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
from network_variant.CSA_RP import CSA_RP
from network_variant.Dual_CA_Softmax import Dual_CA_Softmax
from params import *
import numpy as np
import logzero
from logzero import logger

from utils import visualize_data

logzero.logfile(os.path.join(os.environ['RES_PATH'], 'log.txt'), backupCount=3)

'''
Some of the code about DistributedDataParallel is referenced from:
https://github.com/pytorch/examples/tree/master/imagenet
'''


def cleanup(args):
    logger.info(get_log_str(args, 'cleaning up...'))
    if args.multiprocessing_distributed:
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

    # Use a barrier() to make sure that other process loads the model after process 0 saves it.
    # see https://stackoverflow.com/questions/59760328/how-does-torch-distributed-barrier-work
    if args.multiprocessing_distributed:
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


def pretrain_do_forward(args, model, inputs, recon_criterion, device):
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if args.network in ['Dual_CSA', 'Dual_CA_Softmax']:
        outputs = model(inputs['RP'], inputs['FS'])
    elif args.network == 'CSA_RP':
        outputs = model(inputs['RP'])
    elif args.network == 'CSA_FS':
        outputs = model(inputs['FS'])
    loss = sum([recon_criterion(*recon_ori) for recon_ori in outputs['recon_ori']])
    return outputs, loss


def pretrain_do_forward_backward(args, optimizer, model, inputs, recon_criterion, device):
    optimizer.zero_grad()
    outputs, loss = pretrain_do_forward(args, model, inputs, recon_criterion, device)
    loss.backward()
    optimizer.step()
    return outputs, loss


def pretrain(args, model, train_loader, test_loader, train_sampler=None, device=None):
    logger.info(get_log_str(args, f'pretraining...'))
    pretrain_start = timeit.time.perf_counter()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    recon_criterion = nn.MSELoss()

    best_score = inf
    not_improving_step = 0

    for epoch in range(args.start_epoch, args.epochs, ):
        epoch_start = timeit.time.perf_counter()

        accum_train_loss = 0
        accum_test_loss = 0

        # ? in DistributedDataParallel training, the size of train_sampler.dataset is not equal to actual sample number
        n_total_train_samples = 0
        n_total_test_samples = 0

        # ========== train code ==========
        model.train()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        for batch_idx, (RP, FS, _) in enumerate(train_loader):
            batch_start = timeit.time.perf_counter()
            outputs, loss = pretrain_do_forward_backward(args, optimizer, model, {'RP': RP, 'FS': FS}, recon_criterion,
                                                         device)
            n_total_train_samples += len(RP)
            accum_train_loss += loss.item() * len(RP)
            align_width = len(str(len(train_loader)))
            if batch_idx % args.print_freq == 0 or batch_idx == len(train_loader) - 1:
                logger.info(get_log_str(args, f'Train Epoch: {epoch}, '
                                              f'[{batch_idx + 1:>{align_width}}'
                                              f'/{len(train_loader):>{align_width}}], '
                                              f'Loss: {loss.item():.6f}, '
                                              f'batch time: {timeit.time.perf_counter() - batch_start:.2f}s'))
        # ========== end train code ==========

        # ========== eval code ==========
        model.eval()
        with torch.no_grad():
            for RP, FS, _ in test_loader:
                outputs, loss = pretrain_do_forward(args, model, {'RP': RP, 'FS': FS}, recon_criterion, device)
                n_total_test_samples += len(RP)
                accum_test_loss += loss.item() * len(RP)

        train_loss = accum_train_loss / n_total_train_samples
        test_loss = accum_test_loss / n_total_test_samples
        logger.info(get_log_str(args, f'==> ' +
                                f'avg train_loss: {train_loss:.5f}, ' +
                                f'avg test_loss: {test_loss:.5f}, '
                                f'epoch time: {timeit.time.perf_counter() - epoch_start:.2f}s'))
        # ========== end eval code ==========

        # early stop
        if test_loss < best_score:
            not_improving_step = 0
            logger.info(get_log_str(args, f'★ SCORE IMPROVED FROM {best_score:.5f} TO {test_loss:.5f}'))
            try_to_save_model(args, model, os.path.join(args.results_path, 'pretrained_AE.pt'))
            best_score = test_loss
        else:
            not_improving_step += 1
            if not_improving_step > args.patience:
                logger.info(get_log_str(args, f'* best score: {best_score}, '
                                              f'NOT IMPROVING STEP:{not_improving_step}, EARLY STOP!'))
                break
            else:
                logger.info(get_log_str(args, f'* best: {best_score}, NOT IMPROVING STEP: {not_improving_step}'))
        # end early stop

    if args.visualize_emb:
        model = load_saved_model(args, os.path.join(args.results_path, 'pretrained_AE.pt'), model)
        model.eval()
        with torch.no_grad():
            test_embs = []
            test_true_labels = []
            for RP, FS, true_label in test_loader:
                outputs, loss = pretrain_do_forward(args, model, {'RP': RP, 'FS': FS}, recon_criterion, device)
                test_embs.extend(outputs['emb'].cpu().numpy().tolist())
                test_true_labels.extend(true_label.tolist())
        visualize_data(np.array(test_embs), test_true_labels, N_CLASS,
                       os.path.join(args.results_path, 'visualization', 'pretrained_emb.png'))

    logger.info(get_log_str(args, f'END PRETRAINING ON NODE {args.node}, '
                                  f'RANK: {args.rank}, TIME: {timeit.time.perf_counter() - pretrain_start:.2f}s'))


def joint_training_do_forward(args, model, inputs, recon_criterion, cls_criterion, loss_weight, device):
    inputs = {k: v.to(device) for k, v in inputs.items()}
    alpha, beta, gamma = loss_weight

    if args.network in ['Dual_CSA', 'Dual_CA_Softmax']:
        outputs = model(inputs['RP'], inputs['FS'])
    elif args.network == 'CSA_RP':
        outputs = model(inputs['RP'])
    elif args.network == 'CSA_FS':
        outputs = model(inputs['FS'])

    # note: recon_loss is a list
    recon_loss = [recon_criterion(*recon_ori) for recon_ori in outputs['recon_ori']]
    cls_loss = cls_criterion(torch.log(outputs['pred']).float(), inputs['true_label']) if args.network == 'Dual_CSA' \
        else cls_criterion(outputs['pred'].float(), inputs['true_label'].argmax(dim=1, keepdim=True).view(-1))

    if args.network in ['Dual_CSA', 'Dual_CA_Softmax']:
        # dual AE, dual recon loss
        combine_loss = alpha * recon_loss[0] + beta * cls_loss + gamma * recon_loss[1]
    else:
        # single AE, single recon loss
        combine_loss = 1 * recon_loss[0] + beta * cls_loss
    # note: recon_loss is a list
    return outputs, recon_loss, cls_loss, combine_loss


def joint_train_do_forward_backward(args, optimizer, model, inputs, recon_criterion, cls_criterion, loss_weight,
                                    device):
    optimizer.zero_grad()
    # note: recon_loss is a list
    outputs, recon_loss, cls_loss, combine_loss = joint_training_do_forward(args, model, inputs, recon_criterion,
                                                                            cls_criterion, loss_weight, device)
    combine_loss.backward()
    optimizer.step()
    # note: recon_loss is a list
    return outputs, recon_loss, cls_loss, combine_loss


def joint_train(args, model, train_loader, test_loader, train_sampler=None, loss_weight=None, device=None):
    logger.info(get_log_str(args, f'joint training..., loss weight: {loss_weight}'))
    joint_train_start = timeit.time.perf_counter()

    model.module.set_pretrained(True) if args.multiprocessing_distributed else model.set_pretrained(True)
    recon_criterion = nn.MSELoss()
    # batchmean, Considered carefully
    cls_criterion = nn.KLDivLoss(reduction='batchmean') if args.network in ['Dual_CSA', 'CSA_RP',
                                                                            'CSA_FS'] else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)  # weight_decay added

    best_score = 0
    not_improving_step = 0

    for epoch in range(args.start_epoch, args.epochs, ):
        epoch_start = timeit.time.perf_counter()

        n_train_pred_correct = 0
        n_test_pred_correct = 0

        accum_train_loss = 0
        accum_test_loss = 0

        # ? in DistributedDataParallel training, the size of train_sampler.dataset is not equal to actual sample number
        n_total_train_samples = 0
        n_total_test_samples = 0

        # ========== train code ==========
        model.train()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        for batch_idx, (RP, FS, true_label) in enumerate(train_loader):
            batch_start = timeit.time.perf_counter()
            outputs, recon_loss, cls_loss, combine_loss = \
                joint_train_do_forward_backward(args, optimizer, model,
                                                {'RP': RP, 'FS': FS, 'true_label': true_label}, recon_criterion,
                                                cls_criterion, loss_weight, device)
            accum_train_loss += combine_loss.item() * len(RP)
            pred = outputs['pred'].argmax(dim=1, keepdim=True)
            true_label = true_label.argmax(dim=1, keepdim=True)
            n_train_pred_correct += pred.eq(true_label).sum().item()
            n_total_train_samples += len(RP)
            align_width = len(str(len(train_loader)))
            if batch_idx % args.print_freq == 0 or batch_idx == len(train_loader) - 1:
                logger.info(get_log_str(args, f'Train Epoch: {epoch}, '
                                              f'[{batch_idx + 1:>{align_width}}'
                                              f'/{len(train_loader):>{align_width}}], '
                                              f'recon_loss: {[los.item() for los in recon_loss]:.6f}, '  # list
                                              f'cls_loss: {cls_loss.item():.6f}, '
                                              f'combine_loss: {combine_loss.item():.6f}, '
                                              f'batch time: {timeit.time.perf_counter() - batch_start:.2f}s'))
        # ========== end train code ==========

        # ========== eval code ==========
        model.eval()
        with torch.no_grad():
            test_embs = []
            test_true_labels = []
            for batch_idx, (RP, FS, true_label) in enumerate(test_loader):
                outputs, recon_loss, cls_loss, combine_loss = \
                    joint_training_do_forward(args, model, {'RP': RP, 'FS': FS, 'true_label': true_label},
                                              recon_criterion, cls_criterion, loss_weight, device)
                test_embs.extend(outputs['emb'].cpu().numpy().tolist())
                accum_test_loss += combine_loss.item() * len(RP)
                pred = outputs['pred'].argmax(dim=1, keepdim=True)
                true_label = true_label.argmax(dim=1, keepdim=True)
                test_true_labels.extend(true_label.tolist())
                n_test_pred_correct += pred.eq(true_label).sum().item()
                n_total_test_samples += len(RP)

        # TODO: KLD batchmean loss should not be calculated in accumulate way. here for easy display loss info
        train_loss = accum_train_loss / n_total_train_samples
        test_loss = accum_test_loss / n_total_test_samples
        train_acc = n_train_pred_correct / n_total_train_samples
        test_acc = n_test_pred_correct / n_total_test_samples
        logger.info(get_log_str(args, f'==> ' +
                                f'avg train_loss: {train_loss:.5f}, '
                                f'train acc: {train_acc:.5f}, '
                                f'avg test_loss: {test_loss:.5f}, '
                                f'test_acc: {test_acc:.5f}, '
                                f'epoch time: {timeit.time.perf_counter() - epoch_start:.2f}s'))
        # ========== end eval code ==========

        # visualize latent space
        if args.visualize_emb and epoch % args.visualize_emb == 0:
            visualize_data(np.array(test_embs), test_true_labels, N_CLASS,
                           os.path.join(args.results_path, 'visualization', f'joint_training_emb_epoch{epoch}.png'))

        # early stop
        if test_acc > best_score:
            not_improving_step = 0
            logger.info(get_log_str(args, f'★ SCORE IMPROVED FROM {best_score:.5f} TO {test_acc:.5f}'))
            try_to_save_model(args, model, os.path.join(args.results_path, f'{args.network}.pt'))
            best_score = test_acc
        else:
            not_improving_step += 1
            if not_improving_step > args.patience:
                logger.info(get_log_str(args, f'* best score: {best_score}, '
                                              f'NOT IMPROVING STEP: {not_improving_step}, EARLY STOP!'))
                break
            else:
                logger.info(get_log_str(args, f'* best: {best_score}, NOT IMPROVING STEP: {not_improving_step}'))

    logger.info(get_log_str(args, f'END JOINT TRAINING ON NODE {args.node}, '
                                  f'RANK {args.rank}, TIME: {timeit.time.perf_counter() - joint_train_start:.2f}s'))


def test_predict(args, model, test_loader, loss_weight=None, device=None):
    logger.info(get_log_str(args, f'test_predict..., loss weight: {loss_weight}'))
    predict_start = timeit.time.perf_counter()

    model.eval()
    model.module.set_pretrained(True) if args.multiprocessing_distributed else model.set_pretrained(True)
    recon_criterion = nn.MSELoss()
    cls_criterion = nn.KLDivLoss(reduction='batchmean') \
        if args.network in ['Dual_CSA', 'CSA_RP', 'CSA_FS'] else nn.CrossEntropyLoss()

    with torch.no_grad():
        n_test_pred_correct = 0
        n_total_test_samples = 0
        accum_test_loss = 0
        pred_labels = []
        true_labels = []
        concat_embs = []
        for batch_idx, (RP, FS, true_label) in enumerate(test_loader):
            outputs, recon_loss, cls_loss, combine_loss = \
                joint_training_do_forward(args, model, {'RP': RP, 'FS': FS, 'true_label': true_label},
                                          recon_criterion, cls_criterion, loss_weight, device)

            concat_embs.extend(outputs['emb'].cpu().numpy().tolist())
            accum_test_loss += combine_loss.item() * len(RP)
            pred = outputs['pred'].argmax(dim=1, keepdim=True)
            pred_labels.extend(pred.tolist())
            true_label = true_label.argmax(dim=1, keepdim=True)
            true_labels.extend(true_label.tolist())
            n_corrects = pred.eq(true_label).sum().item()
            n_test_pred_correct += n_corrects
            n_total_test_samples += len(RP)

        test_loss = accum_test_loss / n_total_test_samples
        test_acc = n_test_pred_correct / n_total_test_samples
        logger.info(get_log_str(args, f'==> ' +
                                f'test_loss: {test_loss:.5f}, '
                                f'test_acc: {test_acc:.5f}'))
        logger.info(get_log_str(args, f'END TEST PREDICTING ON NODE {args.node}, '
                                      f'RANK {args.rank}, TIME: {timeit.time.perf_counter() - predict_start:.2f}s'))
        if args.visualize_emb:
            visualize_data(np.array(concat_embs), true_labels, N_CLASS,
                           os.path.join(args.results_path, 'visualization', 'emb_optimal.png'))
        show_classification_results(args, pred_labels, true_labels)


def show_classification_results(args, y_pred, y_true, label_names=['walk', 'bike', 'bus', 'driving', 'train/subway']):
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % args.ngpus_per_node == 0):
        logger.info(get_log_str(args, 'show_classification_results:'))
        cm = confusion_matrix(y_true, y_pred, labels=modes_to_use)
        re = classification_report(y_true, y_pred, target_names=label_names,
                                   digits=5)
        logger.info(get_log_str(args, f'confusion_matrix:\n{str(cm)}'))
        logger.info(get_log_str(args, f'classification_report:\n{str(re)}'))
        with open(os.path.join(args.results_path, 'classification_results.txt'), 'a') as f:
            print(cm, file=f)
            print(re, file=f)


def main_worker(proc, ngpus_per_node, args):
    # note: here args.rank is not adjusted to the global rank among all the processes yet,
    # it represents the machine/node id.
    node = args.rank
    args.node = node
    logger.info(get_log_str(args, f'********THIS IS NODE {node}, PROCESS {proc}********'))
    logger.info(get_log_str(args, f'entering main_worker, process: {proc}, pid: {os.getpid()}'))
    gpu = proc  # process id, i.e., gpu

    if args.seed is not None:
        # for REPRODUCIBILITY see: https://pytorch.org/docs/stable/notes/randomness.html
        # and for issue of DDP training, see：
        # https://stackoverflow.com/questions/62097236/how-to-set-random-seed-when-it-is-in-distributed-training-in-pytorch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        logger.warning(get_log_str(args, f'You have chosen to seed training, seed: {args.seed}'
                                         'This will turn on the CUDNN deterministic setting, '
                                         'which can slow down your training considerably! '
                                         'You may see unexpected behavior when restarting '
                                         'from checkpoints.'))
    else:
        cudnn.benchmark = True

    # init distributed data parallel
    args.gpu = gpu
    if args.gpu is not None:
        logger.info(get_log_str(args, f"Node: {node}: use GPU: {args.gpu} for training"))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # ADJUST RANK! For multiprocessing distributed training, rank needs to be the
            # global rank among [ALL THE PROCESSES]
            args.rank = args.rank * ngpus_per_node + gpu
        logger.info(get_log_str(args, f'Node: {node}: init_process_group, '
                                      f'world_size: {args.world_size}, rank among ALL THE PROCESSES: {args.rank}'))
        # now here world_size means global process number, rank is the ID of current process
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    device = 'cpu'
    centroid = torch.from_numpy(np.load(f'./data/{args.dataset}_features/pedcc.npy'))
    if args.network == 'Dual_CSA':
        model = Dual_CSA(args.n_features, args.RP_emb_dim, args.FS_emb_dim, centroid)
    elif args.network == 'Dual_CA_Softmax':
        model = Dual_CA_Softmax(args.n_features, args.RP_emb_dim, args.FS_emb_dim)
    elif args.network == 'CSA_RP':
        model = CSA_RP(args.n_features, args.RP_emb_dim, centroid)
        assert args.RP_emb_dim == centroid.shape[1]
    elif args.network == 'CSA_FS':
        model = CSA_RP(args.n_features, args.FS_emb_dim, centroid)
        assert args.FS_emb_dim == centroid.shape[1]

    if not torch.cuda.is_available():
        logger.warning(get_log_str(args, 'using CPU, this will be slow'))
        device = 'cpu'
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            logger.info(get_log_str(args, f'set_device: {args.gpu}'))
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
        model.cuda(args.gpu)
        device = f'cuda:{args.gpu}'
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
        device = f'cuda'
    logger.info(get_log_str(args, f'device: {device}'))

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

    # start training or evaluating
    if args.evaluate:
        model = load_saved_model(args, os.path.join(args.results_path, f'{args.network}.pt'), model)
        test_predict(args, model, test_loader, [args.alpha, args.beta, args.gamma], device)
        cleanup(args)
        return

    if args.pretrained:
        # if pretrained, carry on joint training.
        # here we also load Dual_CSA.pt to continue joint training Dual_CSA (if existed)
        if os.path.exists(os.path.join(args.results_path, f'{args.network}.pt')):
            logger.info(get_log_str(args, f'{args.network}.pt exists, continue joint training from last time!'))
            model = load_saved_model(args, os.path.join(args.results_path, f'{args.network}.pt'), model)
        else:
            # load best pretrained model to joint train
            model = load_saved_model(args, os.path.join(args.results_path, 'pretrained_AE.pt'), model)
        joint_train(args, model, train_loader, test_loader, train_sampler, [args.alpha, args.beta, args.gamma], device)
    else:
        # not pretrained, carry on pretrain and joint training in order.
        # here we also load pretrained_AE.pt to continue pretraining (if existed)
        if os.path.exists(os.path.join(args.results_path, 'pretrained_AE.pt')):
            logger.info(get_log_str(args, f'pretrained_AE.pt exists, continue pretraining from last time!'))
            model = load_saved_model(args, os.path.join(args.results_path, 'pretrained_AE.pt'), model)
        pretrain(args, model, train_loader, test_loader, train_sampler, device)
        # load best pretrained model to joint train
        model = load_saved_model(args, os.path.join(args.results_path, 'pretrained_AE.pt'), model)
        if not args.only_pre:
            joint_train(args, model, train_loader, test_loader, train_sampler, [args.alpha, args.beta, args.gamma],
                        device)
    if not args.only_pre:
        # load best Dual_CSA model to predict
        model = load_saved_model(args, os.path.join(args.results_path, f'{args.network}.pt'), model)
        test_predict(args, model, test_loader, [args.alpha, args.beta, args.gamma], device)
    cleanup(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCSA_Training')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--network', type=str, default='Dual_CSA')
    parser.add_argument('--results-path', default='./results/default', type=str)
    parser.add_argument('--alpha', default=ALPHA, type=float)
    parser.add_argument('--beta', default=BETA, type=float)
    parser.add_argument('--gamma', default=GAMMA, type=float)
    parser.add_argument('--RP-emb-dim', type=int)
    parser.add_argument('--FS-emb-dim', type=int)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--no-pre', action='store_true')
    parser.add_argument('--no-joint', action='store_true')
    parser.add_argument('--only-pre', action='store_true')
    parser.add_argument('--no-save-model', action='store_true')
    parser.add_argument('--visualize-emb', type=int, default=0)
    parser.add_argument('--n-features', type=int, default=N_CLASS)

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
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=4, type=int,
                        metavar='N', help='print frequency (default: 4)')
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
    logger.info('args: ' + str(args))

    # write some params to args
    if args.gpu is not None:
        logger.warning('You have chosen a specific GPU. This will completely '
                       'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()  # gpu number of CURRENT MACHINE (this node)

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the TOTAL WORLD_SIZE
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        args.ngpus_per_node = ngpus_per_node
        logger.info(f'ngpus_per_node: {ngpus_per_node}, ACTUAL world_size: {args.world_size}')
        logger.info(f'updated args: {args}')
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function. (nprocs=ngpus_per_node means ONE process for each gpu)
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        logger.info(f'updated args: {args}')
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
