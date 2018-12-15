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

def perturbation( model, x, y, epsilon = 0.25, weight = 0):
    x.requires_grad = True
    loss = model.loss_fn(x,y)
    x_grads = torch.autograd.grad(loss, x)
    x_grad = torch.cat([g.view(-1) for g in x_grads])
    perturbed_ex = weight*x + epsilon
    x.requires_grad = False
    return perturbed_ex

def get_hessian( h, x, y, idx, tr = False):
    grad, loss = h.get_loss_gradient(x, y)
    
    if tr == False:
        filename = "../data/hp_inv_test_" + str(idx) +".npz"
    else:
        filename = "../data/hp_inv_tr_" + str(idx) +".npz"
   
    my_file = Path(filename)

    if my_file.is_file():
        print( "Loading hvp inverse from file {}".format(filename))
        hvp = np.load(filename)["h"]
    else:
        v = tr_grad.detach().numpy()
        print("Calculating HVP inverse for type training {} idx {} ".format(tr,idx))
        hvp = h.get_inverse_hvp_cg(v, max_iterations = 100)
        np.savez(filename, h = hvp)
        
    return hvp



def influence( model, X_train, y_train, X_test, y_test, n_test_indices=5, num_to_remove=5, 
    max_inf = False, n_max_inf = 10, random_seed=7, verify_influence = False, params_change = False):
    
    torch.manual_seed(random_seed)
    N = model.num_train_examples
    predicted = model.predict(X_test)
    test_indices = torch.nonzero(predicted != y_test)
    h = hess.Hessian( model, X_train, y_train )
    h.initialize(X_train, y_train)

    tr_idx = 0
    
    for i in range(n_test_indices):
        test_idx = test_indices[i].item()  
        X_tr, y_tr, X_te, y_te = split_data( tr_idx, test_idx, X_train, y_train, X_test, y_test)
        
        hvp = get_hessian( h, X_te, y_te, test_idx)
    
        if( max_inf ):
            predicted_loss_diffs = np.zeros(n_max_inf)
            for j in range(n_max_inf):
                tr_idx = j
                x_tr, y_tr, x_te, y_te = split_data( tr_idx, test_idx, X_train, 
                                                y_train, X_test, y_test)
                
                tr_grad, tr_loss = h.get_loss_gradient(x_tr, y_tr)
                influence = np.sum(hvp * tr_grad.detach().numpy())/N
                predicted_loss_diffs[j] = influence
            
            np.savez("../data/all_influence_" + str(test_idx) +".npz", p = predicted_loss_diffs)
            indices_to_remove = np.argsort(np.abs(predicted_loss_diffs))[-num_to_remove:]
            predicted_final = predicted_loss_diffs[indices_to_remove]
        else:
            predicted_final = np.zeros(num_to_remove)
            indices_to_remove = np.random.choice(N, num_to_remove)
            for j in range(num_to_remove):
                tr_idx = indices_to_remove[j]
                x_tr, y_tr, x_te, y_te = split_data( tr_idx, test_idx, X_train, 
                                                y_train, X_test, y_test)
                
                tr_grad, tr_loss = h.get_loss_gradient(x_tr, y_tr)
                influence = np.sum(hvp * tr_grad.detach().numpy())/N
                predicted_final[j] = influence
        
        if( verify_influence):
            actual_loss_diff = np.zeros(num_to_remove) 
            print(num_to_remove)
            print(predicted_final.shape)
            print(predicted_loss_diffs.shape)
            print(indices_to_remove.shape)
            for k in range(num_to_remove):
                number = np.array(range(N))
                number = np.delete(number, indices_to_remove[k])

                #device = torch.device("cuda:0")
                #print(device)
                cls_leave = LogitReg(model.iterations, model.D_in, model.D_out, N, model.weight_decay)
                #if device:
                    #cls_leave.to(device)
                cls_leave.fit(X_train[number,:], y_train[number])
                loss = cls_leave.loss_fn(x_te, y_te) - model.loss_fn(x_te, y_te)
                actual_loss_diff[k] = loss
                print( "Predicted loss {} actual loss {}".format(predicted_final[k], actual_loss_diff[k]))

            np.savez("../data/loss_diffs_" + str(test_idx) +".npz",r = {"predicted_loss":predicted_final, "actual_loss": actual_loss_diff})

def influence_k_leave_out( model, X_train, y_train, X_test, y_test, random = True, k = 10, random_seed = 7, n_trials=10, n_test_indices = 1 ):
    torch.manual_seed(random_seed)
    N = model.num_train_examples
    predicted = model.predict(X_test)
    test_indices = torch.nonzero(predicted != y_test)
    tr_idx = 0
    
    for i in range(n_test_indices):
        test_idx = test_indices[i].item()  
        X_tr, y_tr, X_te, y_te = split_data( tr_idx, test_idx, X_train, y_train, X_test, y_test)
        
        h = hess.Hessian( model, X_train, y_train )
        h.initialize(X_train, y_train)
        test_grad, test_loss = h.get_loss_gradient(X_te, y_te)

        print( "norm of gradients {}".format( torch.norm(h.grad) ))
        print( "norm of params {}".format( torch.norm(model.fc1.weight)))

        filename = "../data/hp_inv" + str(test_idx) +".npz"
        my_file = Path(filename)
        if my_file.is_file():
            print( "Loading hvp inverse from file {}".format(filename))
            hvp = np.load(filename)["h"]
        else:
            v = test_grad.detach().numpy()
            print("Calculating HVP inverse")
            hvp = h.get_inverse_hvp_cg(v, max_iterations = 100)
            np.savez(filename, h = hvp)
        
        if(random):
            result = np.zeros((n_trials,2))
            for count in range(n_trials):
                tr_idx = np.random.choice(N,k)
                x_tr, y_tr, x_te, y_te = split_data_k( tr_idx, test_idx, X_train, 
                                                y_train, X_test, y_test)
                tr_grad, tr_loss = h.get_loss_gradient(x_tr, y_tr)
                influence = np.sum(hvp * tr_grad.detach().numpy())/N
                result[count,0] = influence

                number = np.array(range(N))
                number = np.delete(number, tr_idx )
                cls_leave = LogitReg(model.iterations, model.D_in, model.D_out, N, model.weight_decay)
                cls_leave.fit(X_train[number,:], y_train[number])
                loss = cls_leave.loss_fn(x_te, y_te) - model.loss_fn(x_te, y_te)
                result[count,1] = loss
                print( "Trial {} k {} Predicted loss {} actual loss {}".format( count, k, result[count,0], result[count,1] ))
                
            
        np.savez("../data/loss_diffs_" + str(k) + "_" + str(test_idx) +".npz",r = {"predicted_loss":result[:,0],
                                                                      "actual_loss":result[:,1]})
        
        
def influence_perurbation(model, X_train, y_train, X_test, y_test, num_to_remove=5, random_seed=7, 
verify_influence = False, epsilon = 0.25, weight = 0, load_refresh=False):
    print( "Epsilon: {}\n".format(epsilon))
    test_idx = 8
    N = model.num_train_examples
    indices_to_remove = np.random.choice(N, num_to_remove)
    actual_params_diff = np.zeros(num_to_remove)
    actual_loss_diff = np.zeros(num_to_remove)
    predicted_params_diff = np.zeros(num_to_remove)
    predicted_loss_diff =  np.zeros(num_to_remove)
    original_labels = np.zeros((num_to_remove, 5))
    perturbed_labels = np.zeros((num_to_remove, 5))
    K = 100
    
    tr_indices = []
    
    for i in range(num_to_remove):
        
        orig_label = 1
        pred_label = 1
        while( orig_label == pred_label ):
            np.random.seed()
            tr_idx = np.random.choice(N,1)
            X_tr, y_tr, X_te, y_te = split_data( tr_idx, test_idx, X_train, y_train, X_test, y_test)
            h = hess.Hessian( model, X_train, y_train )
            h.initialize(X_train, y_train)
            train_grad, train_loss = h.get_loss_gradient(X_tr, y_tr)

            X_pert = perturbation( model, X_tr, y_tr, epsilon=epsilon, weight=weight )
            
            orig_label = model.predict(X_tr)
            pred_label = model.predict(X_pert)
        
        if( tr_idx in tr_indices):
            continue
            
        tr_indices.append(tr_idx)
        if( i % K == 0):
            print("original_prediction {} original label {}".format( orig_label, y_tr))
            print("perturbed_prediction {} original label {}".format( pred_label, y_tr))
        original_labels[i, 0] = orig_label
        original_labels[i, 1] = torch.max(model.forward(X_tr))
        original_labels[i, 2] = y_tr
        
        perturbed_labels[i,0] = pred_label
        perturbed_labels[i,1] = torch.max(model.forward(X_pert))
        perturbed_labels[i,2] = y_tr
       
        if( i % K == 0):
            print("original_image")
            plt.imshow(X_tr.reshape(28,28))
            plt.show()

            print("Perturbed_image")
            plt.imshow( X_pert.detach().numpy().reshape(28,28))
            plt.show()
            

        perturb_train_grad, pert_loss = h.get_loss_gradient(X_pert, y_tr)

        #diff_v = train_grad
        diff_v = perturb_train_grad - train_grad

        filename = "../data/perturbation/hp_inv_pert" + str(tr_idx) + "_" + str(epsilon) + "_" + str(weight) +".npz"
        my_file = Path(filename)

        if my_file.is_file() and load_refresh == False:
            if( i % K == 0):
                print( "Loading hvp inverse from file {}".format(filename))
            hvp = np.load(filename)["h"][0]
            
        else:
            v = diff_v.detach().numpy()
            #print("Calculating HVP inverse")
            hvp = h.get_inverse_hvp_cg(v, max_iterations = 100)
            np.savez(filename, h = hvp)

        ans = np.linalg.norm(np.array(hvp)/N)
        #print("Answer {}".format(ans))
        predicted_params_diff[i] = ans
        #print(predicted_params_diff[i])
        predicted_loss_diff[i] = np.sum(hvp * train_grad.detach().numpy())/N

        if( verify_influence):
            cls_leave = LogitReg(model.iterations, model.D_in, model.D_out, N, model.weight_decay)
            X = X_train
            
            if( torch.sum(X_pert) == 0):
                number = np.array(range(N))
                number = np.delete(number, indices_to_remove[i])
                X = X[number,:]
                y = y_train[number]
            else:
                if( indices_to_remove[i] >= N ):
                    print(indices_to_remove[i])
                    
                X[indices_to_remove[i], :] = X_pert.detach()
                y = y_train
            
            print(X.shape)
            cls_leave.fit( X.detach(), y)
            
            retrained_orig = cls_leave.predict(X_tr)
            retrained_perturb = cls_leave.predict(X_pert)
            if( i % K == 0):
                print("original_prediction on retrained model {} original label {}", retrained_orig, y_tr)
                print("perturbed_prediction on retrained model {} original label {}", retrained_perturb, y_tr)
                print("Saving_Results at i {}".format(i))
                np.savez("../data/perturbation/loss_diffs_perturb_" + str(epsilon) + "_" +".npz",r = { "idx": tr_indices, 
                                                    "orig_labels":original_labels,
                                                    "pert_labels": perturbed_labels,"predicted_loss":predicted_params_diff, 
                                                    "actual_loss": actual_params_diff})
                
                  
            
            original_labels[i,3] = retrained_orig
            original_labels[i,4] = torch.argmax(cls_leave.forward(X_tr))
            perturbed_labels[i,3] = retrained_perturb
            perturbed_labels[i,4] = torch.argmax(cls_leave.forward(X_pert))
        
            actual_difference = cls_leave.fc1.weight.view(-1) - model.fc1.weight.view(-1)
            actual_params_diff[i] = np.linalg.norm(actual_difference.detach().numpy())
            actual_loss_diff[i] = cls_leave.loss_fn(X_tr, y_tr) - model.loss_fn(X_tr, y_tr)
            
            if( i % K == 0):
                print( "Predicted params diff {} actual params diff {}".format(predicted_params_diff[i], actual_params_diff[i]))
                print( "Predicted loss diff {} actual loss diff {}".format(predicted_loss_diff[i], actual_loss_diff[i]))
    print("Saving_Results at i {}".format(i))
    np.savez("../data/perturbation/loss_diffs_perturb_" + str(epsilon) + "_" +".npz",r = { "idx": tr_indices, 
                                                    "orig_labels":original_labels,
                                                    "pert_labels": perturbed_labels,"predicted_loss":predicted_params_diff, 
                                                    "actual_loss": actual_params_diff}) 