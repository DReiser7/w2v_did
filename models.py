import transformers
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import torch.nn as nn

class Wav2Vec2ClassificationModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.model = Wav2Vec2Model(config)
        
        self.inner_dim = 128
        self.feature_size = 999
        
        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(1024, self.inner_dim)
        self.linear2 = nn.Linear(self.inner_dim*self.feature_size, 5)
        self.init_weights()
        
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

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.model(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = self.linear1(outputs[0]) 
        x = self.tanh(x)
        x = self.linear2(x.view(-1, self.inner_dim*self.feature_size))
        return {'logits':x}