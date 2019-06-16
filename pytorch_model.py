"""Project Doc String - Python program tempalte"""

# imports
from __future__ import print_function
from bass_util import prettyPrint, clog
import json
import os

# Importing PyTorch tools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch import cuda

# Importing other libraries
import numpy as np
import matplotlib.pyplot as plt
import time
# custom classes and functions


class CNN(nn.Module):

    def __init__(self, in_channels=1, out_channels=10, optimizer=optim.SGD, lr=0.01, use_cuda=None, model_name=''):

        # Basics
        super(CNN, self).__init__()
        self.model_name = model_name.split('.')[0]
        self.results_path = 'results'
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        # Settings
        self.optim_type = optimizer
        self.optimizer = None
        self.lr = lr

        # Use CUDA?
        self.use_cuda = use_cuda if (use_cuda != None and cuda.is_available()) else cuda.is_available()
        self.device = 'cpu' if (not self.use_cuda) else ('cuda:'+str(cuda.current_device()))
        self.device = torch.device(self.device)
        clog('Model CUDA:', self.use_cuda, '| Device:', self.device)

        # Current loss and loss history
        self.train_loss = 0
        self.valid_loss = 0
        self.train_loss_hist = []
        self.valid_loss_hist = []

        # Initializing all layers
        self.conv1 = nn.Conv2d(in_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # self.fc1 = nn.Linear(800, 500) # mnist
        self.fc1 = nn.Linear(186050, 500) # cs md
        self.fc2 = nn.Linear(500, out_channels)

        # Running startup routines
        self.startup_routines()

    def startup_routines(self):
        self.optimizer = self.optim_type(self.parameters(), lr=self.lr)
        if self.use_cuda:
            self.cuda()

    def predict(self, input):
        # Switching off autograd
        with torch.no_grad():
            # Use CUDA?
            if self.use_cuda:
                input = input.cuda()
            # Running inference
            return self(input)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def fit_step(self, training_loader, epoch, n_epochs, show_progress=False):

        # Preparations for fit step
        self.train_loss = 0  # Resetting training loss
        self.train()        # Switching to autograd

        for batch_idx, (data, target) in enumerate(training_loader):
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()
            # Clearing gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self(data)
            # Calculating loss
            loss = F.cross_entropy(output, target)
            self.train_loss += loss.item()  # Adding to epoch loss

            # Backward pass and optimization
            loss.backward()                      # Backward pass
            self.optimizer.step()                # Optimizing weights

            if show_progress:
                if batch_idx % int(len(training_loader)*0.10) == 0:
                # if batch_idx % 1 == 0:
                    print('Train Epoch: {}/{} [{:06d}/{:06d} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch+1,
                        n_epochs,
                        batch_idx*len(data), 
                        len(training_loader.dataset),
                        100. * batch_idx / len(training_loader), 
                        loss))

        # Adding loss to history
        self.train_loss_hist.append(self.train_loss / len(training_loader))

    def validation_step(self, validation_loader, show_progress=False):
        self.eval()
        # Preparations for validation step
        self.valid_loss = 0  # Resetting validation loss
        correct = 0
        # Switching off autograd
        with torch.no_grad():

            # Looping through data
            for input, target in validation_loader:

                # Use CUDA?
                if self.use_cuda:
                    input = input.cuda()
                    target = target.cuda()

                # Forward pass
                output = self.forward(input)

                # Calculating loss
                loss = F.cross_entropy(output, target, reduction='sum')
                self.valid_loss += loss.item()  # Adding to epoch loss

                # accuracy
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            self.valid_loss /= len(validation_loader.dataset)

            # Adding loss to history
            self.valid_loss_hist.append(self.valid_loss)

        if show_progress:
            print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                self.valid_loss, 
                correct, 
                len(validation_loader.dataset),
                100. * correct / len(validation_loader.dataset)
                ))

    def fit(self, training_loader, validation_loader=None, epochs=2, show_progress=True, save_best=False, save_plot=False):

        # Helpers
        best_validation = 1e5

        # Looping through epochs
        for epoch in range(epochs):
            self.fit_step(training_loader, epoch, epochs, show_progress)  # Optimizing
            if validation_loader != None:  # Perform validation?
                # Calculating validation loss
                self.validation_step(validation_loader, show_progress)

            # Possibly saving model
            if save_best:
                if self.valid_loss_hist[-1] < best_validation:
                    self.save('best_validation_'+str(epoch))
                    best_validation = self.valid_loss_hist[-1]

        # Switching to eval
        self.eval()

        # save loss to file
        self.save_loss()

        # save plot
        if save_plot:
            self.plot_loss()

    def save_loss(self):
        path = self.results_path + '/' + self.model_name + '_loss_data.json'
        clog('Saving Loss to file:', path)
        data = dict()
        data['train_loss'] = self.train_loss_hist
        data['valid_loss'] = self.valid_loss_hist

        with open(path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=2)
        
    def load_loss(self):
        path = self.results_path + '/' + self.model_name + '_loss_data.json'
        clog('Loading Loss from file:', path)
        with open(path, 'r') as jfile:
            jdata = json.loads(jfile.read())
            return jdata['train_loss'], jdata['valid_loss']

    def save(self, path='cnn_model.pth'):
        if not '.pth' in path:
            path += '.pth'
        path = self.results_path+'/'+ self.model_name + '_' + path
        clog('Saving model: {}'.format(path))
        # torch.save(self.state_dict(), path) # Normal save
        torch.save(self, path) # For visualizing - need the whole model

    def plot_loss(self, plot_name='loss_plot'):
        plot_name = self.model_name + '_' + plot_name
        if not '.png' in plot_name:
            plot_name += '.png'

        # OLD version
        plt.figure()

        # Adding plots
        plt.plot(self.train_loss_hist, color='blue', label='Training loss')
        plt.plot(self.valid_loss_hist, color='red', label='Validation loss')

        # Axis labels
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Pytorch Model Training and Validation loss ['+self.model_name+']')
        plt.legend(loc='upper right')

        # saving plot
        path = self.results_path+'/'+plot_name
        clog('Saving loss plot: {}'.format(path))
        plt.savefig(path)

    @staticmethod
    def load(train=False, path='results/cnn_model.pth'):
        if not '.pth' in path:
            path += '.pth'

        model = CNN()
        model.load_state_dict(torch.load(path))
        if train:
            model.startup_routines()
        return model


# main funciton
def main():
    clog('main')


# run
if __name__ == '__main__':
    main()
