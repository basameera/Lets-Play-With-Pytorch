"""Sudoku Custom Dataset"""

import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split
import os
from PIL import Image
import numpy as np
import pandas as pd
import time


def readCSVfile(path):
    data = pd.read_csv(path)
    data = data[['quizzes', 'solutions']].values
    x, t = data[0, 0], data[0, 1]
    xd, td = [], []
    for n in range(len(x)):
        xd.append(int(x[n]))
        td.append(int(t[n]))

    xd, td = np.array(xd).reshape((1, 9, 9)), np.array(td).reshape((1, 9, 9))
    print(xd.shape)


class datasetFromCSV_2D(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.data = self.data[['quizzes', 'solutions']].values

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x, t = self.data[index, 0], self.data[index, 1]
        xd, td = [], []
        for n in range(len(x)):
            xd.append(int(x[n]))
            td.append(int(t[n]))
        xd, td = np.array(xd).reshape(
            (1, 9, 9)), np.array(td).reshape((1, 9, 9))
        xd, td = torch.tensor(xd, dtype=torch.float), torch.tensor(
            td, dtype=torch.float)
        return xd, td  # x, target

# main funciton


def main():
    # Pytorch Dataset
    print('Before Norm data ================================================')


# run
if __name__ == '__main__':
    main()
