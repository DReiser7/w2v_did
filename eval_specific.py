import csv
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio

from models import Wav2VecClassifierModelMean6 as Wav2VecClassifierModel
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
            self.model = Wav2VecClassifierModel.from_pretrained(dir).to("cuda")
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

        dialects = [
            'us',
            'australia',
            'canada',
            'england',
            'indian',
            'scotland']

        softmax = torch.nn.Softmax(dim=-1)
        probs = softmax(outputs['logits'])
        top_prob, top_lbls = torch.topk(probs[0], 2)
        return {"x": dialects[top_lbls[0]], dialects[top_lbls[0]]: format(float(top_prob[0]), '.2f')}


if __name__ == "__main__":

    data_path = "/cluster/home/fiviapas/en-accents/test/"
    pathlist = Path(data_path).glob('**/*.mp3')
    csv_path = "/cluster/home/fiviapas/data_english/eval_english_5000_2.csv"

    classifier = SpeechClassification(path="/cluster/home/fiviapas/data_english/model-saves/train-accents/2/5000/")

    with open(csv_path, 'w', newline='') as csvfile:
        for path in pathlist:
            label = path.parts[len(path.parts) - 2]
            prediction = classifier.classify(path)

            if label != prediction["x"]:
                print(prediction)
                print(str(path))
                spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow([prediction['x'], prediction[prediction['x']], str(path)])
