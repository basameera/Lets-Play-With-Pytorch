"""Lets solve Sudoku - https://www.kaggle.com/bryanpark/sudoku 
Training of Sudoku NNs
"""
import sys
import os
# path to the custom module
sys.path.append(r'C:\Users\Sameera\Documents\Github\Lets-Play-With-Pytorch')
# 
from utils import cmdArgs, init_torch_seeds
from SkunkWork.utils import clog, getSplitByPercentage, prettyPrint, model_summary
from dataset import datasetFromCSV_2D
from model import sudokuCNN, CNN_SS
import Trainer as swt
from torch.utils.data import DataLoader, random_split
# 
import torch
import torch.nn as nn
import torch.optim as optim
from torch import cuda
# 
import time
import datetime
now = datetime.datetime.now()

def main():
    
    # 1. cmd args
    args = cmdArgs()
    prettyPrint(args.__dict__, 'cmd args')
    
    # cuda settings
    use_cuda = not args.no_cuda and cuda.is_available()
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    init_torch_seeds(use_cuda, seed=0)

    # 2. settings
    settings = dict()
    settings['use cuda'] = use_cuda
    settings['device'] = 'cpu' if (not use_cuda) else (
        'cuda:'+str(cuda.current_device()))
    settings['device'] = torch.device(settings['device'])
    settings['kwargs'] = kwargs

    settings['in_channels'] = 1
    settings['out_channels'] = 9

    settings['save_model'] = True

    prettyPrint(settings, 'settings')

    path_models = 'models/'
    path_data = 'data/'
    path_results = 'results/'

    # 3. data loading
    # path_data += 'sudoku/sudoku_small.csv'
    path_data += 'sudoku/sudoku_half.csv'

    custom_dataset = datasetFromCSV_2D(path_data)

    percentage = 0.9

    clog('Dataset split radio (train, validation, test):',
          getSplitByPercentage(percentage, len(custom_dataset)))

    train_dataset, val_dataset, test_dataset = random_split(
        custom_dataset, getSplitByPercentage(percentage, len(custom_dataset)))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              **kwargs
                             )
    valid_loader = DataLoader(dataset=val_dataset,
                              batch_size=args.valid_batch_size,
                              shuffle=True,
                              **kwargs
                              )
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=32,
                             shuffle=True,
                             **kwargs
                             )

    clog('Data Loaders ready')

    # 4. Model
    model = CNN_SS(
        in_channels=settings['in_channels'], out_channels=settings['out_channels'])

    if settings['use cuda']:
        model.cuda()

    

    path_models += model.__class__.__name__ + '_' + str(datetime.datetime.now()).split('.')[0].replace(':', '-') + '/'
    path_results += model.__class__.__name__ + '_' + str(datetime.datetime.now()).split('.')[0].replace(':', '-') + '/'
    
    # 4.1 Loading model
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
    
    clog('Model ready')
    input_size = (1, 9, 9)
    clog('Model Summary')
    model_summary(model, input_size)
    # print(model.eval())

    # 5. Trainer
    trainer = swt.nnTrainer(
        model=model, model_name=model.__class__.__name__, use_cuda=settings['use cuda'], path_results=path_results)
    # optimizer = optim.SGD(model.parameters(), lr=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    trainer.compile(optimizer, criterion=nn.CrossEntropyLoss(),
                    valid_criterion=nn.CrossEntropyLoss())  # reduction='mean'

    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    clog('Model Total Trainable parameters: {}'.format(pytorch_total_params))

    # 5.1 Trainning model
    if args.train:
        if not os.path.exists(path_models): os.makedirs(path_models)
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

    # 6. Save model
    if args.train and settings['save_model']:
        full = False
        if args.save_model == 'f':
            full = True
        trainer.saveModel(path='model_'+str(args.epochs), full=full)

    # 7. Evaluation/Testing model
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
    main()
