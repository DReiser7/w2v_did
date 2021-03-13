import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
import numpy as np


class ExampleDataset(Dataset):
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

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # format the file path and load the file
        path = self.root_dir + str(self.folders[idx]) + "/" + self.file_names[idx]

        speech, fs = sf.read(path)
        # sound = self.transform(path)

        return speech, self.labels[idx]

    def transform(self, path):
        dict = {}
        count = 0
        for block in sf.blocks(path, blocksize=160000, overlap=16000, fill_value=0):
            dict[count] = block
            count += 1

        data = list(dict.items())
        array = np.array(data)

        return torch.from_numpy(array)
