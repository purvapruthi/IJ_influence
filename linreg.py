from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torch.autograd as autograd
from torchvision import datasets, transforms
import numpy as np


# LeNet Model definition
class LinearReg(nn.Module):
    def __init__(self, iterations):
        super(LinearReg, self).__init__()
        self.fc1 = nn.Linear(2, 1)
        self.iterations = iterations
        
    def forward(self, X):
        h1_output = self.fc1(X)
        #m = nn.LogSoftmax(dim=0)
        #print(h1_output)
        #print(h1_output)
        return h1_output
    
    def loss_fn(self, X, y):
        output = self.forward(X)
        loss = F.mse_loss(output, y)
        print( "loss {}".format(loss))
        return loss
        
    def fit(self, X, y):
        def closure():
            loss = self.loss_fn(X,y)
            self.optimizer.zero_grad()
            loss.backward()
            return loss
        
        self.optimizer = optim.LBFGS(self.parameters(), max_iter = 500, lr=1.5)
        self.optimizer.step(closure)
    
    def predict(self, X):
        y = self.forward(X)
        return y