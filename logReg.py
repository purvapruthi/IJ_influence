from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torch.autograd as autograd
from torchvision import datasets, transforms
import numpy as np
from torch.nn import Parameter
from sklearn import linear_model

# LeNet Model definition
class LogitReg(nn.Module):
    def __init__(self, max_iter, D_in, D_out, N, weight_decay, random_seed = 7):
        torch.manual_seed(random_seed)
        super(LogitReg, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.fc1 = nn.Linear(D_in, D_out, bias=False)
        self.iterations = max_iter
        self.num_train_examples = N  
        self.weight_decay = weight_decay
        
    def forward(self, X):
        h1_output = self.fc1(X)
        m = nn.LogSoftmax(dim=1)
        return m(h1_output)
    
    def loss_fn(self, X, y):
        output = self.forward(X)
        loss = F.nll_loss(output, y)
        return loss
        
    def fit(self, X, y, max_iter = 100, verbose = True):
        num_train_examples = X.shape[0]

        C = 1.0 / (num_train_examples * self.weight_decay) 
        self.model = linear_model.LogisticRegression(
            C=C,
            tol=1e-8,
            fit_intercept=False, 
            solver='lbfgs',
            multi_class='multinomial',
            warm_start=True, #True
            max_iter= self.iterations)


        self.model.fit(X, y)
        w = np.array(self.model.coef_)
       
        self.set_model_params(w)

        if verbose:
            print('LBFGS training took %s iter.' % self.model.n_iter_)
            print('After training with LBFGS: ')
            
    
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

    def set_model_params( self, w ):
        w = torch.from_numpy(w)
        w = w.float()
        self.fc1.weight = Parameter(w, requires_grad = True)

    def get_all_params(self ):
        params = list(self.parameters())
        w1 = np.transpose(params[0].detach().numpy())

        return w1

        