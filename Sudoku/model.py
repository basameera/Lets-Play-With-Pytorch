"""Contain All Models for Sudoku.
In descending order
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_SS(nn.Module):
    """Use semantic segmentation techniques to get a probability output, which indicate
    the relenace of values (1 to 9) for each position.
    """

    def __init__(self, in_channels=1, out_channels=1):

        # Basics
        super(CNN_SS, self).__init__()
        self.version = 'MT3'
        # Initializing all layers
        self.conv1 = nn.Conv2d(in_channels, 20, 3)
        self.conv2 = nn.Conv2d(20, 50, 3)
        self.conv3 = nn.Conv2d(50, 100, 3)

        self.deconv1 = nn.ConvTranspose2d(100, 50, 3)
        self.deconv2 = nn.ConvTranspose2d(50, 20, 3)
        self.deconv3 = nn.ConvTranspose2d(20, out_channels, 3)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        # print('conv1:', x.shape)

        x = F.relu(self.conv2(x))
        # print('conv2:', x.shape)

        x = F.relu(self.conv3(x))
        # print('conv3:', x.shape)

        x = F.relu(self.deconv1(x))
        # print('conv3:', x.shape)

        x = F.relu(self.deconv2(x))
        # print('conv3:', x.shape)

        x = F.softmax(self.deconv3(x), dim=1)
        # print('conv3:', x.shape)
        # raise NotImplementedError
        return x

class sudokuCNN(nn.Module):
    """Model Type 2 (MT2)

    CNN
    """

    def __init__(self, in_channels=1, out_channels=1):

        # Basics
        super(sudokuCNN, self).__init__()
        self.version = 'MT2'
        # Initializing all layers
        self.conv1 = nn.Conv2d(in_channels, 20, 3)
        self.conv2 = nn.Conv2d(20, 50, 3)
        self.conv3 = nn.Conv2d(50, 100, 3)

        self.deconv1 = nn.ConvTranspose2d(100, 50, 3)
        self.deconv2 = nn.ConvTranspose2d(50, 20, 3)
        self.deconv3 = nn.ConvTranspose2d(20, out_channels, 3)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        # print('conv1:', x.shape)

        x = F.relu(self.conv2(x))
        # print('conv2:', x.shape)

        x = F.relu(self.conv3(x))
        # print('conv3:', x.shape)

        x = F.relu(self.deconv1(x))
        # print('conv3:', x.shape)

        x = F.relu(self.deconv2(x))
        # print('conv3:', x.shape)

        x = self.deconv3(x)
        # print('conv3:', x.shape)
        # raise NotImplementedError
        return x


class sudokuModel(nn.Module):
    """Model Type 1 (MT1)

    Linear Model
    """

    def __init__(self, in_channels=1, out_channels=10):

        # Basics
        super(sudokuModel, self).__init__()
        self.version = 'MT1'
        # Initializing all layers
        self.fc1 = nn.Linear(in_channels, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, out_channels)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


if __name__ == "__main__":

    model = sudokuCNN()
    print(model.__doc__)
