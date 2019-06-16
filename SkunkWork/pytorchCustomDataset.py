"""Pytorch Custom Dataset"""

'''
To Do
* Resize images
* without split - two folders for training and test data is already supplied
'''

# imports

# custom classes and functions




import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split
import os
from PIL import Image
import numpy as np
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
            tempList = self.getListOfFiles(path+'/'+key)
            _class = key
            if int_classes:
                _class = value

            for file in tempList:
                self.data_list.update({file: _class})

            self.fileList += tempList

        # image transformations
        self.size = size # TODO : check if tuple and adjust properly
        self.init_transforms = transforms.Compose([
            transforms.Resize(size=(self.size, self.size)),
            transforms.ToTensor(),
        ])

        if self.norm_data:
            if norm_mean is not None and norm_std is not None:
                self.norm_transforms = transforms.Compose([
                    transforms.Normalize(mean=norm_mean, std=norm_std),
                ])
            else:
                raise ValueError("Arguments 'norm_mean' and 'norm_std' vectors are not available.")

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

    def getDatasetSizeOnDisk(self):
        """Get Dataset Size on Local Disk"""
        size = 0
        ext = 'Bytes'
        for file in self.fileList:
            size += os.stat(file).st_size

        if size > 1e6:
            size /= 1e6
            ext = 'MB'
        if size > 1e3:
            size /= 1e3
            ext = 'kB'
        return size, ext

    def getListOfFiles(self, dirName, file_extentions=['.jpg', '.JPG']):
        # create a list of file and sub directories
        # names in the given directory
        listOfFile = os.listdir(dirName)
        allFiles = list()
        # Iterate over a4ll the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(dirName, entry)
            # If entry is a directory then get the list of files in this directory
            if os.path.isdir(fullPath):
                allFiles = allFiles + self.getListOfFiles(fullPath)
            else:
                if os.path.isfile(fullPath) and (fullPath.endswith('.jpg') or fullPath.endswith('.JPG')):
                    allFiles.append(fullPath)

        return allFiles

    def getClasses(self):
        return self.classes

    def getInvClasses(self):
        return self.inverse_classes

    def getSplitByPercentage(self, train_percentage=0.8):
        if train_percentage > 0.0 and train_percentage < 1.0:
            train_p = int(train_percentage*self.__len__())
            valid_p = (self.__len__() - train_p)//2
            return [train_p, valid_p, self.__len__() - train_p - valid_p]
        else:
            raise ValueError('Value should be between 0 and 1.')

# main funciton


def main():
    # Pytorch Dataset
    print('Before Norm data ================================================')
    data_folder_path = 'data/MNIST'
    custom_dataset = ImageClassDatasetFromFolder(data_folder_path, int_classes=True, norm_data=False)
    print(len(custom_dataset))
    print(custom_dataset.getClasses())
    print(custom_dataset.getInvClasses())
    print(custom_dataset.getSplitByPercentage())

    train_dataset, val_dataset, test_dataset = random_split(
        custom_dataset, custom_dataset.getSplitByPercentage(0.8))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=32,
                              shuffle=False)

    print('train')
    train_mean = []
    train_std = []

    for i, image in enumerate(train_loader, 0):
        numpy_image = image[0].numpy()
        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
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
    custom_dataset = ImageClassDatasetFromFolder(data_folder_path, int_classes=True, norm_data=True, norm_mean=train_mean, norm_std=train_std)

    train_dataset, val_dataset, test_dataset = random_split(
        custom_dataset, custom_dataset.getSplitByPercentage(0.8))

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
