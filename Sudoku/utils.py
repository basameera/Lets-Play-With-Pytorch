"""Sudoku Utils"""
import argparse
import torch


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
                        help=r"Type of the loading model, 's': only states, 'f': full model (default: 's') (**Required for loading)")
    parser.add_argument('--lpath', type=str, default='', metavar='',
                        help=r"Path to the loading model. (e.g. 'path\to\model\model_name.pth') (**Required for loading)")

    # save param
    parser.add_argument('--save-model', type=str, default='s', metavar='',
                        help=r"Methods for saving model, 's': only states, 'f': full model (default: 's')")
    parser.add_argument('--save-best', action='store_true', default=False,
                        help='For Saving the current Best Model')
    parser.add_argument('--save-plot', action='store_true', default=True,
                        help='Save the loss plot as .png')
    return parser.parse_args()


def init_torch_seeds(cuda, seed=0):
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    pass
