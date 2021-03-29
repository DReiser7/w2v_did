import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class DidModelHuggingFace(nn.Module):
    def __init__(self, num_classes, exp_norm_func, freeze_fairseq=False):
        super(DidModelHuggingFace, self).__init__()

        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")

        if freeze_fairseq:
            print("Freezing wav2vec layers")
            for params in self.model.base_model.parameters():
                params.requires_grad = False
            for params in self.model.base_model.encoder.parameters():
                params.requires_grad = False
            for params in self.model.base_model.feature_extractor.parameters():
                params.requires_grad = False
            for params in self.model.base_model.feature_projection.parameters():
                params.requires_grad = False
            for params in self.model.encoder.parameters():
                params.requires_grad = False
            for params in self.model.feature_extractor.parameters():
                params.requires_grad = False
            for params in self.model.feature_projection.parameters():
                params.requires_grad = False

        self.classifier_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Linear(256, num_classes)
        )

        self.exp_norm_func = exp_norm_func

    def freeze_recursive(self, model):
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = False
            self.dfs_freeze(child)

    def forward(self, source):
        x = self.model(source)

        # reduce dimension with mean
        x_reduced = torch.mean(x.last_hidden_state, -2)
        x = self.classifier_layer(x_reduced)
        result = self.exp_norm_func(x, dim=1)

        return result
