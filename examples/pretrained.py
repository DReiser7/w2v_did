# !pip install fairseq
# !pip install torch
import torch
import fairseq

if __name__ == "__main__":

    cp_path = '../models/wav2vec_large.pt' # TODO: https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    model = model[0]
    model.eval()

    wav_input_16khz = torch.randn(1, 10000)
    z = model.feature_extractor(wav_input_16khz)
    c = model.feature_aggregator(z)


    print(model)

    # Freeze all the parameters in the network
    for param in model.parameters():
        param.requires_grad = False