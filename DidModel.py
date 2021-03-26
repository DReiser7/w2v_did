import fairseq
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from fairseq.modules import GradMultiply


class DidModel(nn.Module):
    def __init__(self, model_path, num_classes, freeze_fairseq=False, model_small=True):
        super(DidModel, self).__init__()

        cp_path = ""
        if model_path is None:
            if model_small:
                cp_path = 'data/models/wav2vec_small.pt'
            else:
                cp_path = 'data/models/xlsr_53_56k.pt'
        else:
            cp_path = model_path

        print("Loading model: " + cp_path)
        t = time.time()
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        print("Model loaded - duration: " + str((time.time() - t)))
        self.model = model[0]

        if freeze_fairseq:
            print("Freezing fairseq layers")
            for param in self.model.parameters():
                param.requires_grad = False
        if model_small:
            print("Chose Classifier-Layer for model_small")
            self.classifier_layer = nn.Sequential(
                nn.Linear(256, 256),
                # nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
        else:
            print("Chose Classifier-Layer for XLSR")
            self.classifier_layer = nn.Sequential(
                nn.Linear(768, 512),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.Linear(256, num_classes)
            )

    def forward(self, source, padding_mask=None, mask=True, features_only=False):

        # FAIRSEQ CODE wav2vec.py start
        if self.model.feature_grad_mult > 0:
            features = self.model.feature_extractor(source)
            if self.model.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.model.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.model.feature_extractor(source)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.model.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self.model._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[(torch.arange(padding_mask.shape[0], device=padding_mask.device), output_lengths - 1)] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()

        if self.model.post_extract_proj is not None:
            features = self.model.post_extract_proj(features)

        features = self.model.dropout_input(features)
        unmasked_features = self.model.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.model.input_quantizer:
            q = self.model.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.model.project_inp(features)

        if mask:
            x, mask_indices = self.model.apply_mask(features, padding_mask)
            if mask_indices is not None:
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        x = self.model.encoder(x, padding_mask=padding_mask)

        if features_only:
            return {"x": x, "padding_mask": padding_mask}

        if self.model.quantizer:
            q = self.model.quantizer(y, produce_targets=False)
            y = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]

            y = self.model.project_q(y)

            if self.model.negatives_from_everywhere:
                neg_cands, *_ = self.model.quantizer(unmasked_features, produce_targets=False)
                negs, _ = self.model.sample_negatives(neg_cands, y.size(1))
                negs = self.model.project_q(negs)

            else:
                negs, _ = self.model.sample_negatives(y, y.size(1))

            if self.model.codebook_negatives > 0:
                cb_negs = self.model.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.model.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.model.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                cb_negs = self.model.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.model.project_q(y)

            if self.model.negatives_from_everywhere:
                negs, _ = self.model.sample_negatives(unmasked_features, y.size(1))
                negs = self.model.project_q(negs)
            else:
                negs, _ = self.model.sample_negatives(y, y.size(1))

        x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        if self.model.target_glu:
            y = self.model.target_glu(y)
            negs = self.model.target_glu(negs)

        x = self.model.final_proj(x)
        # x = self.model.compute_preds(x, y, negs)

        ## END FAIRSEQ CODE

        # reduce dimension with mean
        x_reduced = torch.mean(x, -2)
        x = self.classifier_layer(x_reduced)
        softmax = F.softmax(x, dim=1)

        result = {"x": x, "softmax": softmax}

        return result
