import csv
import random
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
import librosa
import numpy as np
import torch
import torchaudio
import wandb

from models import Wav2VecClassifierModelMean7 as Wav2VecClassifierModel
from processors import CustomWav2Vec2Processor


class SpeechClassification:

    def __init__(self, path, window_length, number_of_windows, labels):
        self.model = Wav2VecClassifierModel.from_pretrained(path).to("cuda")
        self.processor = CustomWav2Vec2Processor.from_pretrained(path)
        self.window_length = window_length
        self.number_of_windows = number_of_windows
        self.labels = labels

    def classify(self, wav_file):
        return self.predict(self.load_file_to_data(wav_file),
                            self.model, self.processor)

    def load_file_to_data(self, file, srate=16_000):
        batch = {}
        speech_array, sampling_rate = torchaudio.load(file)
        speech_samples = []
        sample_length = self.window_length * sampling_rate
        stop = 0
        for i in range(self.number_of_windows):
            start = 0 if i == 0 else stop
            stop = start + sample_length
            if start * sampling_rate < len(speech_array[0]):
                speech = speech_array[0].numpy()[start:stop]
                if not (speech == np.array([0])).all():  # skip empty sections
                    speech_samples.append(librosa.resample(np.asarray(speech), sampling_rate, srate))

        batch["speech"] = speech_samples
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
            votes[prediction['x']] = votes[prediction['x']] + 1

        max_value = 0
        max_lbl = ''
        for key, value in votes.items():
            if value > max_value:
                max_value = value
                max_lbl = key
        #pick on random vote if no max vote
        if max_lbl == '':
            max_lbl = random.choice(list(votes.keys()))
        return {'x': max_lbl, 'votes': max_value}


if __name__ == "__main__":
    run = sys.argv[1]
    window_length = int(sys.argv[2])
    number_of_windows = int(sys.argv[3])

    model_path = "/cluster/home/reisedom/data_spanish/model-saves/max-samples/" + str(run)+"/4000"

    data_path = "/cluster/home/reisedom/data/spanish-accents-test-aug/test/"
    pathlist = Path(data_path).glob('**/*.mp3')
    csv_path = "/cluster/home/reisedom/data_spanish/major-vote-eval-" + str(window_length) + "s_run"+str(run)+".csv"

    label_names = ['nortepeninsular',
                'centrosurpeninsular',
                'surpeninsular',
                'rioplatense',
                'caribe',
                'andino',
                'mexicano']

    classifier = SpeechClassification(
        path=model_path,
        window_length=window_length,
        number_of_windows=number_of_windows,
        labels=label_names)

    dict_idx = {'nortepeninsular': 0,
                'centrosurpeninsular': 1,
                'surpeninsular': 2,
                'rioplatense': 3,
                'caribe': 4,
                'andino': 5,
                'mexicano': 6}

    preds = np.array([])
    labs = np.array([])

    wandb.init(name=csv_path)

    with open(csv_path, 'w', newline='') as csvfile:
        for path in pathlist:
            subdir = str(path.parent).replace('\\', '/').replace(data_path, '')
            prediction = classifier.classify(path)

            label = path.parts[len(path.parts) - 2]

            np.append(preds, dict_idx[prediction['x']])
            np.append(labs, dict_idx[label])

            if subdir.find(prediction["x"]) == -1:
                print(prediction)
                print(str(path))
                spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow([prediction['x'], prediction['votes'], str(path)])

        acc = accuracy_score(labs, preds)
        f1 = f1_score(labs, preds, average='macro')

        print("run: " + str(run) + " window_legth:" + window_length)
        print("accuracy: " + str(acc))
        print("f1-score: " + str(f1))
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['accuracy', acc, 'f1-score', f1])

        wandb.log(
            {"conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=labs, preds=preds, class_names=label_names)})
