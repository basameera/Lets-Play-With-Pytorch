"""Pytorch Custom Dataset"""

'''
To Do
* Resize images
* without split - two folders for training and test data is already supplied
'''




import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split
import os
from PIL import Image
import numpy as np
from .utils import getListOfFiles, getSplitByPercentage
import pandas as pd
import numpy as np
import time

class ImageClassDatasetFromFolder(Dataset):
    """From Tutorial - https://totoys.github.io/posts/2019-4-10-what-is-torch.utils.data.Dataset-really/
    More info - https://github.com/utkuozbulak/pytorch-custom-dataset-examples#using-torchvision-transforms"""

    def __init__(self, path, int_classes=False, norm_data=False, norm_mean=None, norm_std=None, size=28):
        self.path = path
        self.int_classes = int_classes
        self.norm_data = norm_data
        cls = sorted(os.listdir(path))
        self.classes = dict()
        for i, c in enumerate(cls):
            self.classes.update({c: i})

        self.inverse_classes = dict()
        for key, val in self.classes.items():
            self.inverse_classes.update({val: key})

        # self.data_list = {'path/filename.ext', <int or str class>}
        self.data_list = dict()
        self.fileList = []
        for key, value in self.classes.items():
            tempList = getListOfFiles(path+'/'+key)
            _class = key
            if int_classes:
                _class = value

            for file in tempList:
                self.data_list.update({file: _class})

            self.fileList += tempList

        # image transformations

        if isinstance(size, int):
            self.size = (size, size)
        if isinstance(size, tuple):
            self.size = size
        self.init_transforms = transforms.Compose([
            transforms.Resize(size=self.size),
            transforms.ToTensor(),
        ])

        if self.norm_data:
            if norm_mean is not None and norm_std is not None:
                self.norm_transforms = transforms.Compose([
                    transforms.Normalize(mean=norm_mean, std=norm_std),
                ])
            else:
                raise ValueError(
                    "Arguments 'norm_mean' and 'norm_std' vectors are not available.")

        # Tensor to PIL
        self.ToPILImage = transforms.Compose([
            transforms.ToPILImage(),
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        file_name = list(self.data_list.keys())[index]
        label = list(self.data_list.values())[index]
        item = Image.open(file_name)
        item = self.init_transforms(item)
        if self.norm_data:
            # print("**norm data")
            item = self.norm_transforms(item)
        return item, label

    def ToPILImage(self, tensor_img):
        # print(isinstance(tensor_img, torch.tensor))
        return self.ToPILImage(tensor_img)

    def getClasses(self):
        return self.classes

    def getInvClasses(self):
        return self.inverse_classes


# template


class customClassTemplate(Dataset):
    def __init__(self, path):
        self.path = path
        cls = sorted(os.listdir(path))
        self.classes = dict()
        for i, c in enumerate(cls):
            self.classes.update({c: i})
        self.data_list = dict()
        for c in cls:
            file_names = sorted(os.listdir(os.path.join(path, c)))
            for file_name in file_names:
                self.data_list.update({file_name: c})

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        file_name = list(self.data_list.keys())[index]
        label = list(self.data_list.values())[index]
        item = Image.open(file_name)
        label = self.classes[label]
        return item, label


# class MyCustomDataset(Dataset):
#     def __init__(self, transforms=None):
#         # stuff

#         self.transforms = transforms

#     def __getitem__(self, index):
#         # stuff


#         if self.transforms is not None:
#             data = self.transforms(data)
#         # If the transform variable is not empty
#         # then it applies the operations in the transforms with the order that it is created.
#         return (img, label)

#     def __len__(self):
#         return count # of how many data(images?) you have

def readCSVfile(path):
    print(path)
    data = pd.read_csv(path)
    print(data.values.shape)
    print(data.head())
    data = data[['quizzes', 'solutions']]
    print(data.values.shape)
    # len = data.values.shape[0]/100
    # df = data.loc[:len]
    # df.to_csv('sudoku_small.csv')


    # index = 10
    # # x, t = data[index, 0], data[index, 1]
    # x = '004083002051004300000096710120800006040000500830607900060309040007000205090050803'  
    # t = '974183652651274389283596714129835476746912538835647921568329147317468295492751863'
    # print('X', x, ' T', t)

    # xd, td = [], []
    # for n in range(len(x)):
    #     xd.append(int(x[n]))
    #     td.append(int(t[n]))
    # xd, td = torch.tensor(xd), torch.tensor(td)
    # print(len(xd))
    # print(td)

class datasetFromCSV(Dataset):
    def __init__(self, path, norm_data=False, norm_mean=None, norm_std=None):
        self.data = pd.read_csv(path)
        self.data = self.data[['quizzes', 'solutions']].values
        self.norm_data = norm_data
        
        if self.norm_data:
            if norm_mean is not None and norm_std is not None:
                self.norm_transforms = transforms.Compose([
                    transforms.Normalize(mean=norm_mean, std=norm_std),
                ])
            else:
                raise ValueError(
                    "Arguments 'norm_mean' and 'norm_std' vectors are not available.")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x, t = self.data[index, 0], self.data[index, 1]
        xd, td = [], []
        for n in range(len(x)):
            xd.append(int(x[n]))
            td.append(int(t[n]))
        xd, td = torch.tensor(xd, dtype=torch.float), torch.tensor(td, dtype=torch.float)
        if self.norm_data:
            xd, td = self.norm_transforms(xd), self.norm_transforms(td)
        return xd, td  # x, target
# main funciton


def main():
    # Pytorch Dataset
    print('Before Norm data ================================================')
    data_folder_path = 'data/MNIST'
    custom_dataset = ImageClassDatasetFromFolder(
        data_folder_path, int_classes=True, norm_data=False)
    print(len(custom_dataset))
    print(custom_dataset.getClasses())
    print(custom_dataset.getInvClasses())
    print(getSplitByPercentage(0.8, len(custom_dataset)))

    train_dataset, val_dataset, test_dataset = random_split(
        custom_dataset, getSplitByPercentage(0.8, len(custom_dataset)))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=32,
                              shuffle=False)

    print('train')
    train_mean = []
    train_std = []

    for i, image in enumerate(train_loader, 0):
        numpy_image = image[0].numpy()
        print('img', numpy_image.shape)
        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        print('batch_mean', batch_mean.shape)
        batch_std = np.std(numpy_image, axis=(0, 2, 3))
        train_mean.append(batch_mean)
        train_std.append(batch_std)

    train_mean = torch.tensor(np.mean(train_mean, axis=0))
    train_std = torch.tensor(np.mean(train_std, axis=0))

    print('Mean:', train_mean.item())
    print('Std Dev:', train_std.item())

    print('After norm data ================================================')
    # train_mean = [0.6097, 0.5079, 0.4260]
    # train_std = [0.2694, 0.2605, 0.2625]
    custom_dataset = ImageClassDatasetFromFolder(
        data_folder_path, int_classes=True, norm_data=True, norm_mean=train_mean, norm_std=train_std)

    train_dataset, val_dataset, test_dataset = random_split(
        custom_dataset, getSplitByPercentage(0.8, len(custom_dataset)))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=32,
                              shuffle=False)
    valid_loader = DataLoader(dataset=val_dataset,
                              batch_size=24,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=24,
                             shuffle=True)

    print('Data Ready')


# run
if __name__ == '__main__':
    main()
