import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

class Wav2VecClassifierModelMean2(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)

        # self.inner_dim = 128
        # self.feature_size = 999

        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 2)
        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

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

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = self.linear1(torch.mean(outputs[0], -2))
        x = self.tanh(x)
        x = self.linear2(x)
        return {'logits': x}

class Wav2VecClassifierModelMean3(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)

        # self.inner_dim = 128
        # self.feature_size = 999

        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 3)
        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

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

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = self.linear1(torch.mean(outputs[0], -2))
        x = self.tanh(x)
        x = self.linear2(x)
        return {'logits': x}


class Wav2VecClassifierModelMean5(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)

        # self.inner_dim = 128
        # self.feature_size = 999

        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 5)
        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

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

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = self.linear1(torch.mean(outputs[0], -2))
        x = self.tanh(x)
        x = self.linear2(x)
        return {'logits': x}


class Wav2VecClassifierModelMean6(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)

        # self.inner_dim = 128
        # self.feature_size = 999

        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 6)
        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

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

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = self.linear1(torch.mean(outputs[0], -2))
        x = self.tanh(x)
        x = self.linear2(x)
        return {'logits': x}


class Wav2VecClassifierModelMean7(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)

        # self.inner_dim = 128
        # self.feature_size = 999

        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 7)
        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

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

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = self.linear1(torch.mean(outputs[0], -2))
        x = self.tanh(x)
        x = self.linear2(x)
        return {'logits': x}


class Wav2VecClassifierModelMean8(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)

        # self.inner_dim = 128
        # self.feature_size = 999

        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 8)
        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

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

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = self.linear1(torch.mean(outputs[0], -2))
        x = self.tanh(x)
        x = self.linear2(x)
        return {'logits': x}
