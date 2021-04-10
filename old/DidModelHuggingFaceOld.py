import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


class DidModelHuggingFaceOld(nn.Module):
    def __init__(self, num_classes, exp_norm_func, freeze_fairseq=False):
        super(DidModelHuggingFaceOld, self).__init__()

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

        self.inner = 128
        self.features = 999

        self.leakyReLu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(1024, self.inner)
        self.fc2 = nn.Linear(self.inner * self.features, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, num_classes)

        self.exp_norm_func = exp_norm_func

    def freeze_recursive(self, model):
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = False
            self.dfs_freeze(child)

    def forward(self, source):
        outputs = self.model(source)

        # reduce dimension with mean
        x = self.fc1(outputs[0])
        x = self.leakyReLu(x)
        x = self.fc2(x.view(-1, self.inner * self.features))
        x = self.leakyReLu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        x = self.fc4(x)

        normalized = self.exp_norm_func(x, dim=1)

        x = F.softmax(x, dim=1)

        result = {"x": x, "normalized": normalized}
        return result
