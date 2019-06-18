"""Lets solve Sudoku - https://www.kaggle.com/bryanpark/sudoku """
# imports
from SkunkWork.utils import prettyPrint, clog, getSplitByPercentage
from SkunkWork.pytorchCustomDataset import readCSVfile, datasetFromCSV, ImageClassDatasetFromFolder
import SkunkWork.Trainer as swt
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

# TODO: make the model - conv2D
class sudokuModel(nn.Module):

    def __init__(self, in_channels=1, out_channels=10):

        # Basics
        super(sudokuModel, self).__init__()

        # Initializing all layers
        self.fc1 = nn.Linear(in_channels, 100)
        self.fc2 = nn.Linear(100, 100)
        # self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, out_channels)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        return self.fc4(x)

# custom classes and functions


def cmdArgs():
    parser = argparse.ArgumentParser(
        description='PyTorch NN\n- by Bassandaruwan')
    batch_size = 64
    valid_batch_size = 32
    epochs = 1
    parser.add_argument('--batch-size', type=int, default=batch_size, metavar='N',
                        help='input batch size for training (default: {})'.format(batch_size))
    parser.add_argument('--valid-batch-size', type=int, default=valid_batch_size, metavar='N',
                        help='input batch size for validating (default: {})'.format(valid_batch_size))
    parser.add_argument('--epochs', type=int, default=epochs, metavar='N',
                        help='number of epochs to train (default: {})'.format(epochs))
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the trained Model')
    parser.add_argument('--save-best', action='store_true', default=False,
                        help='For Saving the current Best Model')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Start Training the model')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Start Evaluating the model')
    parser.add_argument('--load', action='store_true', default=False,
                        help='Load the model')
    parser.add_argument('--show_progress', action='store_true', default=True,
                        help='Show training progress')
    parser.add_argument('--save_plot', action='store_true', default=True,
                        help='Save the loss plot as .png')
    return parser.parse_args()


def main():
    args = cmdArgs()
    prettyPrint(args.__dict__, 'cmd args')

    # Pytorch Dataset
    data_folder_path = 'data/sudoku/sudoku_small.csv'
    custom_dataset = datasetFromCSV(
        data_folder_path, norm_data=False)

    print('Dataset split radio (train, validation, test):',
          getSplitByPercentage(0.8, len(custom_dataset)))

    train_dataset, val_dataset, test_dataset = random_split(
        custom_dataset, getSplitByPercentage(0.8, len(custom_dataset)))

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
                             batch_size=1,
                             shuffle=True,
                             pin_memory=True,
                             num_workers=num_workers)

    # for input, _ in test_loader:
    #     print(input.shape)

    clog('Data Loaders ready')

    # raise NotImplementedError

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

    prettyPrint(settings, 'settings')

    clog('Model Ready')

    model = sudokuModel(
        in_channels=settings['in_channels'], out_channels=settings['out_channels'])
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
        clog("Training time: {} seconds | Device: {}".format(
            time.time() - start_time, settings['device']))
        clog('History', history)

    # save model
    if args.train and args.save_model:
        trainer.saveModel(path='model_'+str(args.epochs), full=False)

    if args.eval:
        # test model
        clog('Prediction Test model')
        output = trainer.predict(test_loader, show_progress=True)


# run
if __name__ == '__main__':
    print('\n')
    clog(__file__)

    # readCSVfile('data/sudoku/sudoku_small.csv')

    main()
