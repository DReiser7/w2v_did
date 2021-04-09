import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

class DidModelHuggingFace(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.model = Wav2Vec2Model(config)

        self.classifier_layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 5)
        )

    def freeze_feature_extractor(self):
        print("Freezing wav2vec layers")
        # self.model.freeze_feature_extractor()
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


    def forward( self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None):

        outputs = self.model(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # reduce dimension with mean
        x_reduced = torch.mean(outputs.last_hidden_state, -2)

        x = self.classifier_layer(x_reduced)

        result = {'logits': x}
        return result
