import csv
import random
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio

from archive.model_com_voice import Wav2Vec2CommVoice10sModel
from processors import CustomWav2Vec2Processor


class SpeechClassification:

    def __init__(self, path, window_length, number_of_windows, labels):
        self.model = Wav2Vec2CommVoice10sModel.from_pretrained(path).to("cuda")
        self.processor = CustomWav2Vec2Processor.from_pretrained(path)

    def classify(self, wav_file):
        return self.predict(self.load_file_to_data(wav_file),
                            self.model, self.processor)

    def load_file_to_data(self, file, srate=16_000):
        batch = {}
        speech_array, sampling_rate = torchaudio.load(file)
        speech_samples = []
        sample_length = self.window_lenth * sampling_rate
        for i in range(self.number_of_windows):
            start = random.randrange(0, (len(speech_array[0]) - sample_length))
            stop = start + sample_length
            speech_samples.append(speech_array[0].numpy()[start:stop])

        batch["speech"] = librosa.resample(np.asarray(speech_array), sampling_rate, srate)
        batch["sampling_rate"] = srate
        return batch

    def predict(self, data, model, processor):

        votes = {}
        for lbl in self.labels:
            votes[lbl] = 0

        features = []
        for speech in data['speech']:
            features.append(processor(speech,
                                      sampling_rate=data["sampling_rate"],
                                      return_tensors="pt"))

        outputs = []
        for feature in features:
            input_values = feature.input_values.to("cuda")
            attention_mask = feature.attention_mask.to("cuda")
            with torch.no_grad():
                outputs.append(model(input_values, attention_mask=attention_mask))

        softmax = torch.nn.Softmax(dim=-1)
        predictions = []
        for output in outputs:
            probabilties = softmax(output['logits'])
            top_prob, top_lbls = torch.topk(probabilties[0], 3)
            predictions.append(
                {"x": self.labels[top_lbls[0]], self.labels[top_lbls[0]]: format(float(top_prob[0]), '.2f')})

        for prediction in predictions:
            votes[prediction['x']] = self.votes[prediction['x']] + 1

        max_value = 0
        max_lbl = ''
        for key, value in votes.items():
            if value > max_value:
                max_value = value
                max_lbl = key
        return {'x': max_lbl, 'votes': max_value}


if __name__ == "__main__":

    data_path = "/cluster/home/fiviapas/en-accents/test/"
    pathlist = Path(data_path).glob('**/*.mp3')
    csv_path = "/cluster/home/fiviapas/data_english/major-vote-eval-3s.csv"

    classifier = SpeechClassification(
        path="/cluster/home/fiviapas/data_english/model-saves/train-accents/1/5000",
        window_length=3,
        number_of_windows=3,
        labels=['us',
               'australia',
               'canada',
               'england',
               'indian',
               'scotland'])

    with open(csv_path, 'w', newline='') as csvfile:
        for path in pathlist:
            subdir = str(path.parent).replace('\\', '/').replace(data_path, '')
            prediction = classifier.classify(path)

            if subdir.find(prediction["x"]) == -1:
                print(prediction)
                print(str(path))
                spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow([prediction['x'], prediction[prediction['x']], str(path)])
