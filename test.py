"""Test Python Stuff"""

print(__file__.split('.py')[0])
import torch

if torch.cuda.is_available():
    print('GPU avaiable')
    DEVICE = 'cuda:'+str(torch.cuda.current_device())
    print(DEVICE)