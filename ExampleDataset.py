import torchaudio
from torch.utils.data import Dataset
import torch
import pandas as pd


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
        sound = torchaudio.load(path)
        # load returns a tensor with the sound data and the sampling frequency (44.1kHz for UrbanSound8K)
        sound_data = sound[0]
        # downsample the audio to ~8kHz
        temp_data = torch.zeros([160000, 1])  # tempData accounts for audio clips that are too short
        if sound_data.numel() < 160000:
            temp_data[:sound_data.numel()] = sound_data[:]
        else:
            temp_data[:] = sound_data[:160000]

        sound_data = temp_data
        sound_formatted = torch.zeros([32000, 1])
        sound_formatted[:32000] = sound_data[::5]  # take every fifth sample of soundData
        sound_formatted = sound_formatted.permute(1, 0)

        return sound_formatted, self.labels[idx]