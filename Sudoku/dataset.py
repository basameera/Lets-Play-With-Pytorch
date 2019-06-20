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
import torch.nn.functional as F


def readCSVfile(path):
    data = pd.read_csv(path)
    data = data[['quizzes', 'solutions']].values
    x, t = data[0, 0], data[0, 1]
    xd, td = [], []
    for n in range(len(x)):
        xd.append(int(x[n]))
        td.append(int(t[n]))

    xd, td = np.array(xd).reshape((1, 9, 9)), np.array(td).reshape((1, 9, 9))
    xd, td = torch.tensor(xd, dtype=torch.float), torch.tensor(
        td, dtype=torch.int)
    
    # td = td - 1
    # print(xd, xd.shape)

    print(td, td.shape)
    input = td.view(9*9)
    
    # input = torch.arange(0, 5)    
    
    print(input.shape)

    OH = F.one_hot(input, num_classes=9)
    print(OH, OH.shape)


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
            td, dtype=torch.long)
        td -= 1
        # print('x', xd.shape, ' t', td.shape)
        return xd, td  # x, target

# main funciton


def main():
    # Pytorch Dataset
    path_data = '../data/'

    # 3. data loading
    path_data += 'sudoku/sudoku_small.csv'
    readCSVfile(path_data)


# run
if __name__ == '__main__':
    main()
