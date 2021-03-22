import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC


class DidModelHuggingFace(nn.Module):
    def __init__(self, num_classes, freeze_fairseq=False):
        super(DidModelHuggingFace, self).__init__()

        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")

        if freeze_fairseq:
            print("Freezing fairseq layers")
            for param in self.model.parameters():
                param.requires_grad = False

        self.classifier_layer = nn.Sequential(
            nn.Linear(32, 512),
            # nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Linear(256, num_classes)
        )

    def forward(self, source, padding_mask=None, mask=True, features_only=False):
        x = self.model(source)

        # reduce dimension with mean
        x_reduced = torch.mean(x.logits, -2)
        x = self.classifier_layer(x_reduced)
        softmax = F.softmax(x, dim=1)

        result = {"x": x, "softmax": softmax}

        return result
