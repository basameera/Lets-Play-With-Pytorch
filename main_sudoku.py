"""Lets solve Sudoku - https://www.kaggle.com/bryanpark/sudoku """
# imports
import os
import sys
# path to the custom module
sys.path.append(r'C:\Users\Sameera\Documents\Github\Lets-Play-With-Pytorch')
from SkunkWork.utils import prettyPrint, clog, getSplitByPercentage
from SkunkWork.pytorchCustomDataset import readCSVfile, datasetFromCSV, ImageClassDatasetFromFolder, datasetFromCSV_2D
from SkunkWork import swTrainer as swt
#
from torch.utils.data import DataLoader, random_split
import torch
from torch import cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#
import argparse
import time
import numpy as np
import datetime

# TODO: make the model - conv2D


class sudokuModel(nn.Module):

    def __init__(self, in_channels=1, out_channels=10):

        # Basics
        super(sudokuModel, self).__init__()

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


class sudokuCNN(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):

        # Basics
        super(sudokuCNN, self).__init__()

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
# custom classes and functions


def cmdArgs():
    parser = argparse.ArgumentParser(
        description='PyTorch NN\n- by Bassandaruwan')
    batch_size = 64
    valid_batch_size = 32
    epochs = 1
    # train param
    parser.add_argument('--batch-size', type=int, default=batch_size, metavar='N',
                        help='input batch size for training (default: {})'.format(batch_size))
    parser.add_argument('--valid-batch-size', type=int, default=valid_batch_size, metavar='N',
                        help='input batch size for validating (default: {})'.format(valid_batch_size))
    parser.add_argument('--epochs', type=int, default=epochs, metavar='N',
                        help='number of epochs to train (default: {})'.format(epochs))
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Start Training the model')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Start Evaluating the model')
    parser.add_argument('--show-progress', action='store_true', default=False,
                        help='Show training progress')

    # load param
    parser.add_argument('--load', action='store_true', default=False,
                        help='Load the model: True/False')
    parser.add_argument('--ltype', type=str, default='s', metavar='',
                        help='Type of the loading model, \'s\': only states, \'f\': full model (default: \'s\') (**Required for loading)')
    parser.add_argument('--lpath', type=str, default='', metavar='',
                        help='Path to the loading model. (e.g. \'path\\to\model\model_name.pth\') (**Required for loading)')

    # save param
    parser.add_argument('--save-model', type=str, default='s', metavar='',
                        help='Methods for saving model, \'s\': only states, \'f\': full model (default: \'s\')')
    parser.add_argument('--save-best', action='store_true', default=False,
                        help='For Saving the current Best Model')
    parser.add_argument('--save-plot', action='store_true', default=True,
                        help='Save the loss plot as .png')
    return parser.parse_args()


def main():
    args = cmdArgs()
    prettyPrint(args.__dict__, 'cmd args')

    # Pytorch Dataset
    # data_folder_path = 'data/sudoku/sudoku_small.csv'
    data_folder_path = 'data/sudoku/sudoku.csv'

    CNN = True
    save_model = True

    if CNN:
        # Conv
        custom_dataset = datasetFromCSV_2D(data_folder_path)
    else:
        # Linear
        custom_dataset = datasetFromCSV(data_folder_path)

    percentage = 0.9

    print('Dataset split radio (train, validation, test):',
          getSplitByPercentage(percentage, len(custom_dataset)))

    train_dataset, val_dataset, test_dataset = random_split(
        custom_dataset, getSplitByPercentage(percentage, len(custom_dataset)))

    num_workers = 4

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=num_workers)
    valid_loader = DataLoader(dataset=val_dataset,
                              batch_size=args.valid_batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=32,
                             shuffle=True,
                             pin_memory=True,
                             num_workers=num_workers)

    clog('Data Loaders ready')

    settings = dict()
    use_cuda = not args.no_cuda and cuda.is_available()

    # reproducibility
    torch.manual_seed(0)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # settings
    settings['use cuda'] = use_cuda
    settings['device'] = 'cpu' if (not use_cuda) else (
        'cuda:'+str(cuda.current_device()))
    settings['device'] = torch.device(settings['device'])

    settings['in_channels'] = 81
    settings['out_channels'] = 81

    if CNN:
        settings['in_channels'] = 1
        settings['out_channels'] = 1

    prettyPrint(settings, 'settings')

    clog('Model Ready')

    if CNN:
        # Conv
        model = sudokuCNN(
            in_channels=settings['in_channels'], out_channels=settings['out_channels'])
    else:
        # Linear
        model = sudokuModel(
            in_channels=settings['in_channels'], out_channels=settings['out_channels'])


    # load model
    '''
    load: true/false
    type: states/full
    path: path to the previously saved model
    '''
    if args.load:
        if args.ltype=='s':
            clog('Loading model with states from: {}'.format(args.lpath))
            model.load_state_dict(torch.load(args.lpath))
        if args.ltype=='f':
            clog('Loading full model from: {}'.format(args.lpath))
            model = torch.load(args.lpath)
        

    print(model.eval())

    trainer = swt.nnTrainer(
        model=model, model_name=__file__, use_cuda=settings['use cuda'])

    trainer.compile(optim.SGD, criterion=nn.MSELoss(),
                    valid_criterion=nn.MSELoss())  # reduction='mean'

    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    clog('Model Total Trainable parameters: {}'.format(pytorch_total_params))

    # Train model
    if args.train:
        clog('Training Started...\n')
        start_time = time.time()
        history = trainer.fit(train_loader, valid_loader, epochs=args.epochs, save_best=args.save_best,
                              show_progress=args.show_progress, save_plot=args.save_plot)
        clog("Training time: {}  |  Device: {}".format(
            str(datetime.timedelta(seconds=(time.time() - start_time))).split('.')[0], 
            settings['device'])
            )
        clog('History', history)

    # save model
    if args.train and save_model:
        full = False
        if args.save_model=='f':
            full = True
        trainer.saveModel(path='model_'+str(args.epochs), full=full)

    if args.eval:
        # test model
        clog('Prediction Test model')
        # output = trainer.predict(test_loader, show_progress=True)
        output = trainer.evaluate(test_loader)

        (P, T) = output[0]

        print(P[0])
        print(T[0])


# run
if __name__ == '__main__':
    print('\n')
    clog(__file__)

    # readCSVfile('data/sudoku/sudoku_small.csv')

    main()
