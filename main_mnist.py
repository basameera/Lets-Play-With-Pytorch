"""Project Doc String - Python program tempalte"""
# imports
from SkunkWork.pytorchCustomDataset import ImageClassDatasetFromFolder
from pytorch_model import CNN
from torch.utils.data import DataLoader, random_split
import torch
from torch import cuda
import argparse
from SkunkWork.utils import prettyPrint, clog, getSplitByPercentage
import time
# custom classes and functions
def cmdArgs():
    parser = argparse.ArgumentParser(
        description='PyTorch NN\n- by Bassandaruwan')
    batch_size = 64
    valid_batch_size = 32
    epochs = 0
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


# main funciton
def main():
    args = cmdArgs()
    prettyPrint(args.__dict__, 'cmd args')

    # Pytorch Dataset
    norm_mean = [0.1349952518939972]
    norm_std = [0.30401742458343506]
    data_folder_path = 'data/MNIST'
    custom_dataset = ImageClassDatasetFromFolder(data_folder_path, int_classes=True, norm_data=True, norm_mean=norm_mean, norm_std=norm_std)
    print('Classes:', custom_dataset.getClasses())
    print('Decode Classes:', custom_dataset.getInvClasses())
    print('Dataset split radio (train, validation, test):', getSplitByPercentage(0.8, len(custom_dataset)))

    train_dataset, val_dataset, test_dataset = random_split(
        custom_dataset, getSplitByPercentage(0.8, len(custom_dataset)))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=False)
    valid_loader = DataLoader(dataset=val_dataset,
                              batch_size=args.valid_batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=True)
    
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
    settings['device'] = 'cpu' if (not use_cuda) else ('cuda:'+str(cuda.current_device()))
    settings['device'] = torch.device(settings['device'])
    settings['in_channels'] = 1
    settings['out_channels'] = 10

    prettyPrint(settings, 'settings')

    clog('Model Ready')
    # Instantiate mode
    model = CNN(in_channels=settings['in_channels'], out_channels=settings['out_channels'], use_cuda=use_cuda, model_name=__file__)
    print()
    print(model.eval())

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    clog('Model Total Trainable parameters: {}'.format(pytorch_total_params))

    # Train model
    if args.train:
        clog('Training Started...\n')
        model.fit(train_loader, valid_loader, epochs=args.epochs, save_best=args.save_best, show_progress=args.show_progress, save_plot=args.save_plot)

    # save model
    if args.train and args.save_model:
        model.save(path='model_'+str(args.epochs))

    # load model
    if args.load:
        clog('Loading model')
        model = CNN().load()

    if args.eval:
        # test model
        clog('Prediction Test model')
        correct_pred = 0
        for i, (input, target) in enumerate(test_loader):
            output = model.predict(input)
            status = ''
            if target[0].item() == torch.argmax(output[0]):
                status = 'Correct'
                correct_pred += 1
            # print('Target: {} | Prediction: {} | status: {}'.format( target[0].item(), torch.argmax(output[0]), status ))

        print('\n\nAccuracy: [{}/{}] - ({:.0f}%)\n'.format( correct_pred, len(test_loader), (correct_pred*100/len(test_loader)) ))

# run
if __name__ == '__main__':
    clog(__file__)
    main()