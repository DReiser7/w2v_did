import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor


class DidDataset(Dataset):
    """ExampleDataset."""

    # Argument List
    #  path to the  csv file
    #  path to the audio Files audio files
    #  list of folders to use in the dataset

    def __init__(self, csv_file, root_dir):
        csv_data = pd.read_csv(csv_file)
        # initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.labels = []
        self.folders = []
        # loop through the csv entries and only add entries from folders in the folder list
        for i in range(0, len(csv_data)):
            self.file_names.append(csv_data.iloc[i, 0])
            self.labels.append(csv_data.iloc[i, 1])
            self.folders.append(csv_data.iloc[i, 2])

        self.root_dir = root_dir

        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                          do_normalize=True, return_attention_mask=True)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # format the file path and load the file
        path = self.root_dir + str(self.folders[idx]) + "/" + self.file_names[idx]

        # speech, fs = sf.read(path)
        # sound = self.transform(path)

        sound = np.load(path)
        out = self.feature_extractor(sound, sampling_rate=16000).input_values[0]

        return out, np.array(self.labels[idx])

    def transform(self, path):
        dict = {}
        count = 0
        for block in sf.blocks(path, blocksize=160000, overlap=16000, fill_value=0):
            dict[count] = block
            count += 1

        data = list(dict.items())
        array = np.array(data)

        return torch.from_numpy(array)
