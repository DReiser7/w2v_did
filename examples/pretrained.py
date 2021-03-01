# !pip install fairseq TODO does not work for me
# !pip install torch
import torch
import fairseq

if __name__ == "__main__":

    cp_path = '../models/wav2vec_vox_960h_pl.pt' # TODO: download https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_960h_pl.pt
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    model = model[0]
    model.eval()

    wav_input_16khz = torch.randn(1, 10000)
    z = model.feature_extractor(wav_input_16khz)
    c = model.feature_aggregator(z)