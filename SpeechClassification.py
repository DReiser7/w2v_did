from model_klaam import Wav2Vec2KlaamModel
from models import Wav2Vec2ClassificationModel
from processors import CustomWav2Vec2Processor
import torch
import librosa

class SpeechClassification:

    def __init__(self, path=None):
        if path is None:
            dir = 'Zaid/wav2vec2-large-xlsr-dialect-classification'
            self.model = Wav2Vec2KlaamModel.from_pretrained(dir).to("cuda")
            self.processor = CustomWav2Vec2Processor.from_pretrained(dir)
        else:
            dir = path
            self.model = Wav2Vec2ClassificationModel.from_pretrained(dir).to("cuda")
            self.processor = CustomWav2Vec2Processor.from_pretrained(dir)

    def classify(self, wav_file):
        return self.predict(self.load_file_to_data(wav_file),
                       self.model, self.processor)

    def load_file_to_data(file, srate=16_000):
        batch = {}
        speech, sampling_rate = librosa.load(file, sr=srate)
        batch["speech"] = speech
        batch["sampling_rate"] = sampling_rate
        return batch

    def predict(data, model, processor):

        max_length = 320000
        features = processor(data["speech"][:max_length],
                             sampling_rate=data["sampling_rate"],
                             max_length=max_length,
                             pad_to_multiple_of=max_length,
                             padding=True, return_tensors="pt")

        input_values = features.input_values.to("cuda")
        attention_mask = features.attention_mask.to("cuda")
        with torch.no_grad():
            outputs = model(input_values, attention_mask=attention_mask)

        dialects = ['EGY', 'NOR', 'GLF', 'LAV', 'MSA']

        softmax = torch.nn.Softmax(dim=-1)
        probs = softmax(outputs['logits'])
        top_prob, top_lbls = torch.topk(probs[0], 5)
        return {dialects[top_lbls[lbl]]: format(float(top_prob[lbl]), '.2f') for lbl in range(5)}