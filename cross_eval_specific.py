import csv
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio

from archive.model_com_voice import Wav2Vec2CommVoice10sModel
from archive.model_com_voice5 import Wav2Vec2CommVoice5Lang10sModel
from archive.model_klaam import Wav2Vec2KlaamModel
from processors import CustomWav2Vec2Processor


class SpeechClassification:

    def __init__(self, path=None):
        if path is None:
            dir = 'Zaid/wav2vec2-large-xlsr-dialect-classification'
            self.model = Wav2Vec2KlaamModel.from_pretrained(dir).to("cuda")
            self.processor = CustomWav2Vec2Processor.from_pretrained(dir)
        else:
            dir = path
            self.model = Wav2Vec2CommVoice10sModel.from_pretrained(dir).to("cuda")
            self.processor = CustomWav2Vec2Processor.from_pretrained(dir)


    def classify(self, wav_file):
        return self.predict(self.load_file_to_data(wav_file),
                       self.model, self.processor)

    def load_file_to_data(self, file, srate=16_000):
        batch = {}
        speech_array, sampling_rate = torchaudio.load(file)
        speech_array = speech_array[0].numpy()[:10 * srate]
        batch["speech"] = librosa.resample(np.asarray(speech_array), sampling_rate, srate)
        batch["sampling_rate"] = srate
        return batch

    def predict(self, data, model, processor):

        max_length = 160000
        features = processor(data["speech"][:max_length],
                             sampling_rate=data["sampling_rate"],
                             max_length=max_length,
                             pad_to_multiple_of=max_length,
                             padding=True, return_tensors="pt")

        input_values = features.input_values.to("cuda")
        attention_mask = features.attention_mask.to("cuda")
        with torch.no_grad():
            outputs = model(input_values, attention_mask=attention_mask)

        dialects = ['nl', 'es', 'it']

        softmax = torch.nn.Softmax(dim=-1)
        probs = softmax(outputs['logits'])
        top_prob, top_lbls = torch.topk(probs[0], 3)
        return {"x": dialects[top_lbls[0]], dialects[top_lbls[0]]: format(float(top_prob[0]), '.2f')}

class SpeechClassification5:

    def __init__(self, path=None):
        if path is None:
            dir = 'Zaid/wav2vec2-large-xlsr-dialect-classification'
            self.model = Wav2Vec2KlaamModel.from_pretrained(dir).to("cuda")
            self.processor = CustomWav2Vec2Processor.from_pretrained(dir)
        else:
            dir = path
            self.model = Wav2Vec2CommVoice5Lang10sModel.from_pretrained(dir).to("cuda")
            self.processor = CustomWav2Vec2Processor.from_pretrained(dir)


    def classify(self, wav_file):
        return self.predict(self.load_file_to_data(wav_file),
                       self.model, self.processor)

    def load_file_to_data(self, file, srate=16_000):
        batch = {}
        speech_array, sampling_rate = torchaudio.load(file)
        speech_array = speech_array[0].numpy()[:10 * sampling_rate]
        batch["speech"] = librosa.resample(np.asarray(speech_array), sampling_rate, srate)
        batch["sampling_rate"] = srate
        return batch

    def predict(self, data, model, processor):

        max_length = 160000
        features = processor(data["speech"][:max_length],
                             sampling_rate=data["sampling_rate"],
                             max_length=max_length,
                             pad_to_multiple_of=max_length,
                             padding=True, return_tensors="pt")

        input_values = features.input_values.to("cuda")
        attention_mask = features.attention_mask.to("cuda")
        with torch.no_grad():
            outputs = model(input_values, attention_mask=attention_mask)

        dialects = ['nl', 'es', 'it', 'ch', 'ru']

        softmax = torch.nn.Softmax(dim=-1)
        probs = softmax(outputs['logits'])
        top_prob, top_lbls = torch.topk(probs[0], 5)
        return {"x": dialects[top_lbls[0]], dialects[top_lbls[0]]: format(float(top_prob[0]), '.2f')}

if __name__ == "__main__":

    data_path = "/cluster/home/fiviapas/data_Europarl/test-converted/wav/"
    pathlist = Path(data_path).glob('**/*.mp3')
    csv_path = "/cluster/home/fiviapas/data_Europarl/eval.csv"

    # classifier = SpeechClassification(path="/cluster/home/fiviapas/data_LID/model-saves/train-comvoice-b-16-s10/")
    classifier = SpeechClassification5(path="/cluster/home/fiviapas/data_LID/model-saves/train-comvoice5lang-b-16-s10/")

    with open(csv_path, 'w', newline='') as csvfile:
        for path in pathlist:
            subdir = str(path.parent).replace('\\', '/').replace(data_path, '')
            prediction = classifier.classify(path)

            if subdir.find(prediction["x"]) == -1:
                print(prediction)
                print(str(path))
                spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow([prediction['x'], prediction[prediction['x']], str(path)])



