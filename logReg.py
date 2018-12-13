from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torch.autograd as autograd
from torchvision import datasets, transforms
import numpy as np
from torch.nn import Parameter

# LeNet Model definition
class LogitReg(nn.Module):
    def __init__(self, iterations, D_in, D_out, N):
        super(LogitReg, self).__init__()
        self.fc1 = nn.Linear(D_in, D_out, bias=False)
        self.iterations = iterations
        self.fc1.weight = Parameter(torch.zeros(D_out,D_in), requires_grad = True)
        C = 1.0 / (self.num_train_examples * self.weight_decay)   
        self.num_train_examples = N     
        
        self.sklearn_model = linear_model.LogisticRegression(
            C=C,
            tol=1e-8,
            fit_intercept=False, 
            solver='lbfgs',
            multi_class='multinomial',
            warm_start=True, #True
            max_iter=max_lbfgs_iter)

        C_minus_one = 1.0 / ((self.num_train_examples - 1) * self.weight_decay)
        self.sklearn_model_minus_one = linear_model.LogisticRegression(
            C=C_minus_one,
            tol=1e-8,
            fit_intercept=False, 
            solver='lbfgs',
            multi_class='multinomial',
            warm_start=True, #True
            max_iter=max_lbfgs_iter) 
        
    def forward(self, X):
        h1_output = self.fc1(X)
        m = nn.LogSoftmax(dim=1)
        return m(h1_output)
    
    def loss_fn(self, X, y):
        output = self.forward(X)
        #print(output)
        loss = F.nll_loss(output, y)
        #print( "loss {}".format(loss))
        return loss
        
    def fit(self, X, y, batch_size = 1400, max_iter = 100):
        optimizer = torch.optim.LBFGS(self.parameters(), lr=0.1, max_iter=100)
        def closure():
            
            optimizer.zero_grad()
            loss = self.loss_fn(X,y)
            loss.backward()
            return loss

                
        optimizer.step(closure)
    
    def predict(self, X):
        y = self.forward(X)
        return torch.argmax(y, dim=1)
    
    def score(self, X, y):
        y_predict = self.predict(X)
        a = torch.sum(y_predict == y).type(torch.FloatTensor)
        b = y.shape[0]
        print("a {} b {}".format(a,b))
        c = a/b
        c.type(torch.FloatTensor)
        return c