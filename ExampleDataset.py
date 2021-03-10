from torch.utils.data import Dataset
import torch
import pandas as pd
import librosa
import soundfile as sf
import numpy as np


class ExampleDataset(Dataset):
    """ExampleDataset."""
    # Argument List
    #  path to the  csv file
    #  path to the audio Files audio files
    #  list of folders to use in the dataset

    def __init__(self, csv_file, root_dir):
        csvData = pd.read_csv(csv_file)
        # initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.labels = []
        self.folders = []
        # loop through the csv entries and only add entries from folders in the folder list
        for i in range(0, len(csvData)):
            self.file_names.append(csvData.iloc[i, 0])
            self.labels.append(csvData.iloc[i, 1])
            self.folders.append(csvData.iloc[i, 2])

        self.root_dir = root_dir

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # format the file path and load the file
        path = self.root_dir + str(self.folders[idx]) + "/" + self.file_names[idx]
        sound = librosa.load(path, sr=16000,  mono=True)

        speech, fs = sf.read(path)
        sound = librosa.resample(speech, fs, 16000)

        return sound, self.labels[idx]
