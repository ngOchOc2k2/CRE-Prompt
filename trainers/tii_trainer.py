import datetime
import time
import torch

from datasets import build_continual_dataloader
from models.backbone import BertRelationEncoder
from models.model import Classifier
from engines.tii_engine import train_and_evaluate


def train(args):
    device = torch.device(args.device)

    data_loader, data_loader_per_cls, class_mask, target_task_map = build_continual_dataloader(args)

    print("Creating original model")
    model = BertRelationEncoder(args).to(device)

    # Freeze the encoder
    for n, p in model.named_parameters():
        p.requires_grad = False

    classifer = Classifier(args).to(device)

    n_parameters = sum(p.numel() for p in classifer.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = torch.optim.Adam(classifer.parameters(), lr=args.encoder_lr)
    lr_scheduler = None
    criterion = torch.nn.CrossEntropyLoss().to(device)


    print(f"Start training for {args.encoder_epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model, classifer,
                       criterion, data_loader, data_loader_per_cls, optimizer, lr_scheduler,
                       device, class_mask, target_task_map, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")