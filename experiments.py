from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torch.autograd as autograd
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from logReg import LogitReg
from util import *
import hessian as hess
import importlib
from pathlib import Path
from torch.nn import Parameter

def stochastic_hessian_Lissa( h, X_train, y_train, v, max_iter = 1, num_samples = 1, depth = 10000, scale = 10, batch_size = 10):
    print_iter = 1000
    N = X_train.shape[0]
    h.initialize(X_train,y_train)
    loss_fn = h.get_fmin_loss_fn(v)
    x = np.zeros_like(v)
    final_estimate = np.zeros_like(v)
    for iter in range(max_iter):
        for i in range(num_samples):
            print(x.shape)
            X_i_0 = h.get_fmin_hvp(x, x) - v
            X_i_prev = X_i_0
            for j in range(depth):
                np.random.seed()
                indices = np.random.choice(N, batch_size)
                X = X_train[indices,:]
                y = y_train[indices]
                h.initialize(X,y)
                X_i_current =  X_i_0 + X_i_prev - h.get_fmin_hvp( x, X_i_prev)/scale 
                X_i_prev = X_i_current

                if (j % print_iter == 0) or (j == depth - 1):
                    print("Recursion at depth %s: norm is %.8lf" % (j, np.linalg.norm(X_i_current)))

            final_estimate = final_estimate + X_i_prev/scale

        final_estimate = final_estimate/num_samples

        x = x - final_estimate
        print("Function value at iter {} and estimate {}".format( iter, loss_fn(x)))
    return x