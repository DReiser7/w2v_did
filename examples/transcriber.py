# !pip install transformers
# !pip install torch
# !pip install soundfilepip insta
import soundfile as sf
import torch
from transformers import Wav2Vec2ForMaskedLM, Wav2Vec2Tokenizer

if __name__ == "__main__":

    # load pretrained model
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForMaskedLM.from_pretrained("facebook/wav2vec2-base-960h")

    # load audio
    audio_input, _ = sf.read("sample1.flac")

    # transcribe
    input_values = tokenizer(audio_input, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]

    print(transcription)