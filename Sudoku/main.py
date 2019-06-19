"""Lets solve Sudoku - https://www.kaggle.com/bryanpark/sudoku 
Training of Sudoku NNs
"""


import datetime
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import cuda
from torch.utils.data import DataLoader, random_split

import SkunkWork.Trainer as swt
from models import sudokuCNN, sudokuModel
from SkunkWork.pytorchCustomDataset import (datasetFromCSV, datasetFromCSV_2D,
                                            readCSVfile)
from SkunkWork.utils import clog, getSplitByPercentage, prettyPrint
from utils import cmdArgs


def main():
    """
    1. cmd args
    2. settings based on cmd args (model, dataset, training, ...)
    3. data loading
    4. model instantiation
    5. training
    6. saving
    7. evaluation / testing of model
    """
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
        if args.ltype == 's':
            clog('Loading model with states from: {}'.format(args.lpath))
            model.load_state_dict(torch.load(args.lpath))
        if args.ltype == 'f':
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
            str(datetime.timedelta(
                seconds=(time.time() - start_time))).split('.')[0],
            settings['device'])
        )
        clog('History', history)

    # save model
    if args.train and save_model:
        full = False
        if args.save_model == 'f':
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
