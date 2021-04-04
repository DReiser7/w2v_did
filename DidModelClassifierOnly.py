import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


class DidModelClassifierOnly(nn.Module):
    def __init__(self, num_classes, exp_norm_func, freeze_fairseq=False):
        super(DidModelClassifierOnly, self).__init__()

        self.classifier_layer = nn.Sequential(
            nn.Linear(160000, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, num_classes)
        )

        self.exp_norm_func = exp_norm_func

    def freeze_recursive(self, model):
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = False
            self.dfs_freeze(child)

    def forward(self, source):

        # reduce dimension with mean

        x = self.classifier_layer(source)

        normalized = self.exp_norm_func(x, dim=1)

        x = F.softmax(x, dim=1)

        result = {"x": x, "normalized": normalized}
        return result
