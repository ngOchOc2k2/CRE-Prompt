import torch.nn as nn
import torch
import torch.nn.functional as F
from .backbone import BertRelationEncoder


class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(args.encoder_output_size * 2, args.encoder_output_size * 2),
            # nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Linear(args.encoder_output_size * 2, args.encoder_output_size * 2),
            # nn.ReLU(inplace=True),
        ).to(args.device)

        self.head =  nn.Linear(args.encoder_output_size * 2, args.num_of_relation).to(args.device)

    def forward(self, x):
        out = self.mlp(x)
        out = out + x
        return self.head(out)



class ClassifierBasic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(args.encoder_output_size * 2, args.rel_per_task * args.num_tasks, bias=True),
        ).to(args.device)

    def forward(self, x):
        return self.head(x)
