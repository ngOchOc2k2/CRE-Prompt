import yaml
import torch

import numpy as np
import random

from config import Param


def run(args):
    print(f"hyper-parameter configurations:")
    print(yaml.dump(args.__dict__, sort_keys=True, indent=4))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    if hasattr(args, 'train_inference_task_only') and args.train_inference_task_only:
        import trainers.tii_trainer as tii_trainer
        tii_trainer.train(args)
    else:
        import trainers.prompt_trainer as prompt_trainer
        prompt_trainer.train(args)


if __name__ == "__main__":
    # Load configuration
    param = Param()
    args = param.args

    # Device
    torch.cuda.set_device(args.gpu)
    args.device = torch.device(args.device)

    # Num GPU
    args.n_gpu = torch.cuda.device_count()

    # Task name
    args.task_name = args.dataname

    # rel_per_task
    args.rel_per_task = 8 if args.dataname == "FewRel" else 4

    # Run
    run(args)
