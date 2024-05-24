import math
import sys
import os
import datetime
import json
import copy
from typing import Iterable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import utils

from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal
from timm.utils import accuracy


def train_and_evaluate(model: torch.nn.Module, classifier: torch.nn.Module,
                       criterion, data_loader: Iterable, data_loader_per_cls: Iterable,
                       optimizer: torch.optim.Optimizer, lr_scheduler,
                       device: torch.device,
                       class_mask=None, target_task_map=None, args=None, ):
    # create matrix to save end-of-task accuracies
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    pre_ca_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    global cls_mean
    global cls_cov
    global old_head
    cls_mean = dict()
    cls_cov = dict()

    for task_id in range(args.num_tasks):

        # Create new optimizer for each task to clear optimizer status
        if task_id > 0:
            optimizer = torch.optim.Adam(classifier.parameters(), lr=args.encoder_lr)
            lr_scheduler = None

        if task_id > 0:
            old_head = copy.deepcopy(classifier)

        for epoch in range(args.encoder_epochs):
            # Train model
            train_stats = train_one_epoch(model=model, classifier=classifier, criterion=criterion,
                                          data_loader=data_loader[task_id]['train'], optimizer=optimizer,
                                          device=device, epoch=epoch, max_norm=args.max_grad_norm,
                                          set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args, )

            if lr_scheduler:
                lr_scheduler.step(epoch)

        print('-' * 20)
        print(f'Evaluate task {task_id + 1} before CA')
        test_stats_pre_ca = evaluate_till_now(model=model, classifier=classifier, data_loader=data_loader,
                                              device=device,
                                              task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                                              acc_matrix=pre_ca_acc_matrix, args=args)
        print('-' * 20)

        # TODO compute mean and variance
        print('-' * 20)
        print(f'Compute mean and variance for task {task_id + 1}')
        _compute_mean(model=model, data_loader=data_loader_per_cls, device=device, class_mask=class_mask[task_id],
                      args=args)
        print('-' * 20)

        # TODO classifier alignment
        if task_id > 0:
            print('-' * 20)
            print(f'Align classifier for task {task_id + 1}')
            train_task_adaptive_prediction(classifier, args, device, class_mask, task_id)
            print('-' * 20)

        # Evaluate model
        print('-' * 20)
        print(f'Evaluate task {task_id + 1} after CA')
        test_stats = evaluate_till_now(model=model, classifier=classifier, data_loader=data_loader,
                                       device=device,
                                       task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                                       acc_matrix=acc_matrix, args=args)
        print('-' * 20)

        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            state_dict = {
                'model': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }

            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir,
                                   '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))),
                      'a') as f:
                f.write(json.dumps(log_stats) + '\n')





def train_one_epoch(model: torch.nn.Module, classifier, criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args=None, ):
    model.train(set_training_mode)
    classifier.train(set_training_mode)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.encoder_epochs)) + 1}}/{args.encoder_epochs}]'

    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(input)
        logits = classifier(output["x_encoded"])

        # here is the trick to mask out classes of non-current tasks
        if class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target)

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, classifier, data_loader,
             device, task_id=-1, class_mask=None, target_task_map=None, args=None, ):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Test: Task[{task_id + 1}]'

    model.eval()
    classifier.eval()
    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(input)
            logits = classifier(output["x_encoded"])

            # # here is the trick to mask out classes of non-current tasks
            # if args.task_inc and class_mask is not None:
            #     mask = class_mask[task_id]
            #     mask = torch.tensor(mask, dtype=torch.int64).to(device)
            #     logits_mask = torch.ones_like(logits, device=device) * float('-inf')
            #     logits_mask = logits_mask.index_fill(1, mask, 0)
            #     logits = logits + logits_mask

            loss = criterion(logits, target)
            
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            metric_logger.update(Loss=loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

            task_id_preds = torch.max(logits, dim=1)[1]
            task_id_preds = torch.tensor([target_task_map[v.item()] for v in task_id_preds]).to(device)
            batch_size = input.shape[0]
            tii_acc = torch.sum(task_id_preds == task_id) / batch_size
            metric_logger.meters['TII Acc'].update(tii_acc.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f} TII Acc {tii_acc.global_avg:.3f}'
        .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'],
                losses=metric_logger.meters['Loss'], tii_acc=metric_logger.meters['TII Acc']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, classifier, data_loader,
                      device, task_id=-1, class_mask=None, target_task_map=None, acc_matrix=None, args=None, ):
    stat_matrix = np.zeros((3, args.num_tasks))  # 3 for Acc@1, Acc@5, Loss
    tii_acc_matrix = np.zeros((args.num_tasks, ))

    for i in range(task_id + 1):
        test_stats = evaluate(model=model, classifier=classifier, data_loader=data_loader[i]['test'],
                              device=device, task_id=i, class_mask=class_mask, target_task_map=target_task_map,
                              args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']
        tii_acc_matrix[i] = test_stats['TII Acc']

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)
    avg_tii_acc = np.divide(np.sum(tii_acc_matrix), task_id + 1)
    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}\tTII Acc: {:.4f}".format(
        task_id + 1,
        avg_stat[0],
        avg_stat[1],
        avg_stat[2],
        avg_tii_acc)
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                              acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats



@torch.no_grad()
def _compute_mean(model: torch.nn.Module, data_loader: Iterable, device: torch.device, class_mask=None, args=None, ):
    model.eval()

    for cls_id in class_mask:
        data_loader_cls = data_loader[cls_id]['train']
        features_per_cls = []
        for i, (inputs, targets) in enumerate(data_loader_cls):
            inputs = inputs.to(device, non_blocking=True)
            features = model(inputs)['x_encoded']
            features_per_cls.append(features)
        features_per_cls = torch.cat(features_per_cls, dim=0)

        cls_mean[cls_id] = features_per_cls.mean(dim=0)
        cls_cov[cls_id] = torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device)


def train_task_adaptive_prediction(model: torch.nn.Module, args, device, class_mask=None, task_id=-1):
    model.train()

    current_classifier = copy.deepcopy(model)


    run_epochs = args.classifier_epochs
    crct_num = 0
    param_list = [p for p in model.parameters() if p.requires_grad]
    print('-' * 20)
    print('Learnable parameters:')
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)
    print('-' * 20)
    network_params = [{'params': param_list, 'lr': args.classifier_lr, 'weight_decay': args.weight_decay}]
    optimizer = optim.SGD(network_params, lr=args.classifier_lr, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for i in range(task_id):
        crct_num += len(class_mask[i])

    # TODO: efficiency may be improved by encapsulating sampled data into Datasets class and using distributed sampler.
    for epoch in range(run_epochs):
        num_sampled_pcls = args.batch_size
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

        sample_input, sample_target, sample_target_mask = sample_data(
            num_sampled_pcls, task_id, class_mask, 
            device, args, include_current_task=True, target_mask=True)
        
        sf_indexes = torch.randperm(sample_input.size(0))
        inputs = sample_input[sf_indexes]
        targets = sample_target[sf_indexes]
        target_mask = sample_target_mask[sf_indexes]


        for _iter in range(crct_num):
            inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            tgt_mask = target_mask[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            outputs = model(inp)
            logits = outputs

            # CE loss
            if class_mask is not None:
                mask = []
                for id in range(task_id+1):
                    mask.extend(class_mask[id])
                # print(mask)
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
            
            loss = criterion(logits, tgt)

            if args.gamma_old > 0 or args.gamma_aux > 0:
                # Distill on old classes
                old_inp = inp[tgt_mask == 0]
                current_inp = inp[tgt_mask == 1]

                # loss_KD = torch.zeros(task_id + 1).to(device)
                loss_KD = torch.zeros(2).to(device)

                if args.gamma_old > 0:
                    # KD loss for previous tasks
                    with torch.no_grad():
                        old_logits = old_head(old_inp)

                
                    if class_mask is not None:
                        mask = []
                        for t in range(task_id):
                            mask.extend(class_mask[t])
                        not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                        old_task_logits = old_logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                        task_logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))[tgt_mask == 0]

                    loss_KD[0] = F.kl_div(F.log_softmax(task_logits[:, mask], dim=1),
                                            F.softmax(old_task_logits[:, mask], dim=1),
                                                reduction='batchmean')


                if args.gamma_aux > 0:
                    # KD loss for current task
                    with torch.no_grad():
                        aux_logits = current_classifier(current_inp)

                    if class_mask is not None:
                        mask = class_mask[task_id]
                        not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                        aux_logits = aux_logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                        task_logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))[tgt_mask == 1]

                    # loss_KD[task_id] = F.kl_div(F.log_softmax(task_logits[:, class_mask[task_id]], dim=1),
                    #                             F.softmax(aux_logits[:, class_mask[task_id]] * beta, dim=1),
                    #                             reduction='batchmean')    
                    loss_KD[1] = F.kl_div(F.log_softmax(task_logits[:, class_mask[task_id]], dim=1),
                                                F.softmax(aux_logits[:, class_mask[task_id]], dim=1),
                                                reduction='batchmean')     
                
                # loss = loss + args.gamma_old * loss_KD[:task_id].sum() + args.gamma_aux * loss_KD[task_id]
                loss = loss + args.gamma_old * loss_KD[0] + args.gamma_aux * loss_KD[1]

            acc1, acc5 = accuracy(logits, tgt, topk=(1, 5))

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=inp.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=inp.shape[0])

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)

        scheduler.step()



def sample_data(num_sampled_pcls, task_id, class_mask, device, args, include_current_task=True, target_mask=False):
    sampled_data = []
    sampled_label = []
    sampled_target_mask = []
    # if train:
    #     num_sampled_pcls = int(args.batch_size / args.nb_classes * args.num_tasks)
    # else:
    #     num_sampled_pcls = args.batch_size
    if include_current_task:
        max_task = task_id + 1
    else:
        max_task = task_id

    for i in range(max_task):
        for c_id in class_mask[i]:
            mean = torch.tensor(cls_mean[c_id], dtype=torch.float64).to(device)
            cov = cls_cov[c_id].to(device)
            m = MultivariateNormal(mean.float(), cov.float())
            sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
            sampled_data.append(sampled_data_single)

            sampled_label.extend([c_id] * num_sampled_pcls)

            if target_mask:
                if i == task_id:
                    sampled_target_mask.extend([1] * num_sampled_pcls)
                else:
                    sampled_target_mask.extend([0] * num_sampled_pcls)


    sampled_data = torch.cat(sampled_data, dim=0).float().to(device)
    sampled_label = torch.tensor(sampled_label).long().to(device)

    if target_mask:
        sampled_target_mask = torch.tensor(sampled_target_mask).long().to(device)
        return sampled_data, sampled_label, sampled_target_mask

    return sampled_data, sampled_label