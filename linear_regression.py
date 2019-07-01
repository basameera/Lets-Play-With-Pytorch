import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
pts = 50

x_vals = np.random.rand(2, 50)
print(x_vals.shape)
x_train = np.asarray(x_vals, dtype=np.float32).reshape(-1, 1)
print(x_train.shape)
m = 1
alpha = np.random.rand(1)
beta = np.random.rand(1)
y_correct = np.asarray([2*i+m for i in x_vals],
                       dtype=np.float32).reshape(-1, 1)


class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(LinearRegressionModel, self).__init__()
        # Calling Super Class's constructor
        self.linear = nn.Linear(input_dim, output_dim)
        # nn.linear is defined in nn.Module

    def forward(self, x):
        # Here the forward pass is simply a linear function

        out = self.linear(x)
        return out


input_dim = 1
output_dim = 1

# create our model just as we do in Scikit-Learn / C / C++//
model = LinearRegressionModel(input_dim, output_dim)

criterion = nn.MSELoss()  # Mean Squared Loss
l_rate = 0.01
# Stochastic Gradient Descent
optimiser = torch.optim.SGD(model.parameters(), lr=l_rate)

epochs = 2000


for epoch in range(epochs):

    epoch += 1
    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_correct))

    # clear grads
    optimiser.zero_grad()
    # forward to get predicted values
    outputs = model.forward(inputs)
    loss = criterion(outputs, labels)
    loss.backward()  # back props
    optimiser.step()  # update the parameters
    # print('epoch {}, loss {}'.format(epoch, loss.item()))


predicted = model.forward(Variable(torch.from_numpy(x_train))).data.numpy()

plt.plot(x_train, y_correct, 'go', label='from data', alpha=.5)
plt.plot(x_train, predicted, label='prediction', alpha=0.5)
plt.legend()
plt.show()
print(model.state_dict())
