import datetime
import time
import torch
import torch.nn as nn

from datasets import build_continual_dataloader
from models.backbone import BertRelationEncoder
from models.model import Classifier, ClassifierBasic
from models.prompt import Prompt
from engines.prompt_engine import train_and_evaluate


def train(args):
    device = torch.device(args.device)

    data_loader, data_loader_per_cls, class_mask, target_task_map = build_continual_dataloader(args)

    print("Creating original model")
    original_model = BertRelationEncoder(args).to(device)
    original_classifier = Classifier(args).to(device)

    print("Creating model")
    classifier = ClassifierBasic(args).to(device)

    # Freeze the encoder
    for n, p in original_model.named_parameters():
        p.requires_grad = False

    # Scale lr to batch size
    args.encoder_lr = args.encoder_lr * args.batch_size / 128.0
    args.classifier_lr = args.classifier_lr * args.batch_size / 128.0
    args.prompt_pool_lr = args.prompt_pool_lr * args.batch_size / 128.0

    # Train classifier and prompt pools
    prompt_pools = nn.ModuleList([Prompt(args) for _ in range(args.num_tasks)]).to(device)

    if args.larger_prompt_lr:
        base_params = list(prompt_pools.parameters())
        classifier_params = list(classifier.parameters())
        base_params = {
            'params': base_params,
            'lr': args.prompt_pool_lr
        }
        classifier_params = {
            'params': classifier_params,
            'lr': args.prompt_pool_lr * 0.1
        }
        optimizer = torch.optim.Adam([base_params, classifier_params])
    else:
        optimizer = torch.optim.Adam(list(prompt_pools.parameters()) + list(classifier.parameters()), lr=args.prompt_pool_lr)


    lr_scheduler = None
    criterion = torch.nn.CrossEntropyLoss().to(device)


    print(f"Start training for {args.prompt_pool_epochs} epochs")
    start_time = time.time()

    train_and_evaluate(original_model, original_classifier, classifier, prompt_pools,
                       criterion, data_loader, data_loader_per_cls, optimizer, lr_scheduler,
                       device, class_mask, target_task_map, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")