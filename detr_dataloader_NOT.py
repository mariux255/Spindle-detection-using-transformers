import torch
from scipy.signal import butter, sosfilt, sosfreqz
import scipy.io
import random
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


class sentdex_data(Dataset):
    def __init__(self, root_dir = '/home/marius/Documents/OneDrive/MSc/StartUP/Code/DREAMER.mat'):
        self.root_dir = root_dir
        mat = scipy.io.loadmat(root_dir)

        dataset = np.squeeze(mat['DREAMER'][0][0][0])
        dataset_cleaned = []
        for i,subject_data in enumerate(dataset):
            dataset_cleaned.append(subject_data[0][0])
        dataset_cleaned = np.asarray(dataset_cleaned)

        self.videos = []
        self.labels = []
        for i,subject_data in enumerate(dataset_cleaned):
            for j,video in enumerate(subject_data[2][0][0][1]):
                self.videos.append(video[0][-128*60:-1,:])
                self.labels.append(torch.tensor([subject_data[4][j][0], subject_data[5][j][0], subject_data[6][j][0]]))
                


    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        target_video = torch.tensor(self.videos[idx]).float()
        target_video = target_video.view(14,-1)
        labels_return = self.labels[idx].type(torch.FloatTensor)
        return (target_video, labels_return)
