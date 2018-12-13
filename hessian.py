from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torch.autograd as autograd
import numpy as np
from torch.autograd import Variable
from torch.nn import Parameter
from pathlib import Path
from scipy.optimize import fmin_ncg

class Hessian(object):
    """
    Multi-class classification.
    """

    def __init__(self, model, X_train, Y_train, X_test,  Y_test, X_leave, Y_leave):
        np.random.seed(0)
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.X_leave = X_leave
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.Y_leave = Y_leave
        self.vec_to_list = self.get_vec_to_list_fn()
        self.num_train_examples = self.X_train.shape[0]

    def initialize(self, X, y):
        self.clear_grad()
        self.loss = self.model.loss_fn(X, y)
        self.grad = self.get_gradients(self.loss)

    def get_inverse_hvp_cg(self, v):
            
            self.initialize(self.X_train, self.Y_train)
            fmin_loss_fn = self.get_fmin_loss_fn(v)
            fmin_grad_fn = self.get_fmin_grad_fn(v)
            cg_callback = self.get_cg_callback(v)

            fmin_results = fmin_ncg(
                f=fmin_loss_fn,
                x0= v,
                fprime=fmin_grad_fn,
                fhess_p= self.get_fmin_hvp,
                callback=cg_callback,
                avextol=1e-8,
                maxiter=8) 

            return self.vec_to_list(fmin_results)


    def get_fmin_grad_fn(self, v):
            def get_fmin_grad(x):
                print("fmin_grad_fn called")
                hessian_vector_val = self.get_hvp(self.vec_to_list(x))
                
                return hessian_vector_val - v
            return get_fmin_grad

    def get_fmin_hvp(self, x, v):
        print("fmin_hvp called")
        print("x {}".format(np.linalg.norm(x)))
        hessian_vector_val = self.get_hvp(self.vec_to_list(v))

        return hessian_vector_val

    def get_hvp(self, v):
        print("get_hvp called")
        v = torch.from_numpy(np.array(v))[0]
        print(v.shape)

        hessian_vector_val = self.hvp_computation(self.grad, v)
        print("HVP_norm {}".format(torch.norm(hessian_vector_val)))
        return hessian_vector_val.detach().numpy()

    def get_fmin_loss_fn(self, v):

        def get_fmin_loss(x):
            print("loss function called")
            hessian_vector_val = self.get_hvp(self.vec_to_list(x))

            return 0.5 * np.dot(hessian_vector_val, x) - np.dot(v, x)
        return get_fmin_loss


    def get_cg_callback(self, v):
            fmin_loss_fn = self.get_fmin_loss_fn(v)
            
            def fmin_loss_split(x):
                hessian_vector_val = self.get_hvp(self.vec_to_list(x))

                return 0.5 * np.dot(hessian_vector_val, x), - np.dot(v, x)

            def cg_callback(x):
                # x is current params
                v = self.vec_to_list(x)
                idx_to_remove = 5
                print( "x {}".format(np.linalg.norm(x)))

                     
                train_grad_loss_val = self.get_gradients(self.model.loss_fn(self.X_leave, self.Y_leave))

                train_grad_loss_val = train_grad_loss_val.detach().numpy()
                predicted_loss_diff = np.dot(v, train_grad_loss_val) / self.num_train_examples

                print("Train_grad_norm {} ".format(np.linalg.norm(train_grad_loss_val)) )
                print('Function value: %s' % fmin_loss_fn(x))
                quad, lin = fmin_loss_split(x)
                print('Split function value: %s, %s' % (quad, lin))
                print('Predicted loss diff %s' % (predicted_loss_diff))

            return cg_callback

    def get_vec_to_list_fn(self):
        
        def vec_to_list(v):
            return_list = []
            cur_pos = 0
            for p in self.model.parameters():
                length = p.view(-1).shape[0]
                return_list.append(v[cur_pos : cur_pos+length])
                cur_pos += length

            assert cur_pos == len(v)
            return return_list

        return vec_to_list

    def actual_hvp_computation(self, grad, param, v):
        g = grad.contiguous().view(-1)
        v = self.vector.contiguous().view(-1)
        hessian = eval_hessian(g, param)
        actual_hvp = hessian @ v
        return actual_hvp

    def clear_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def get_gradients(self, loss):
        grads = autograd.grad(loss, self.model.parameters(), create_graph=True)
        return torch.cat([g.view(-1) for g in grads])

    def get_hessian_product(self, loss, vector, print_output = False, max_iterations = 10 ):
        
        grads = self.get_gradients(loss)
        result[name] = conjugate_gradient_optimization( grads, vector, p, 
                    print_output = print_output, max_iterations = max_iterations )
        return result

    def hvp_computation(self, grad, v, damping = 1e-2):
        result = autograd.grad(torch.sum(grad* Variable(v)), self.model.parameters(), retain_graph=True, create_graph=True)
        r = torch.cat([g.view(-1) for g in result])
        r = r + damping*v
        return r


    def conjugate_gradient_optimization( self, loss, b, max_iterations = 10, print_output = False):
        x = torch.zeros_like(b)
        Ax = hvp_computation(loss, x)
        r_k = Ax - b 
        d_k = -r_k
        
        initial_norm = torch.norm(r_k)

        count = 1
        while (count < max_iterations): 
            
            Ad = hvp_computation(loss, d_k)
            alpha_k = torch.sum(r_k*r_k)/torch.sum(d_k* Ad)
            x = x + alpha_k* d_k

            r_k1 = r_k + alpha_k*Ad
            beta_k1 = torch.sum(r_k1*r_k1)/torch.sum(r_k*r_k)
            d_k1 = -r_k1 + beta_k1*d_k
            r_k = r_k1
            d_k = d_k1
            current_norm = torch.norm(r_k)
            Ax = hvp_computation(loss, x)
     
            K = 1/2 * torch.sum(x * Ax) - torch.sum(b * x) 
            
            if print_output:
                print( "r_norm: {} K: {}".format(current_norm, K))
                print("iteration:", count)
                
            if torch.isclose(r_k, torch.zeros(r_k.shape)).all():
                print("Breaking!")
                break
            
            if(torch.isnan(current_norm)):
                print("Norm Infinite")
                break

            count = count + 1
        return x

    def eval_hessian(self, loss_grad, param):
        l = loss_grad.size(0)
        hessian = torch.zeros(l, l)
        for idx in range(l):
            grad2rd = autograd.grad(loss_grad[idx],param, create_graph=True)
            cnt = 0
            for g in grad2rd:
                g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
                cnt = 1
            hessian[idx] = g2
        return hessian

    def influence_up(self, x_test, y_test, x_tr, y_tr, X_train, Y_train):
        
        #Get derivative w.r.t removed train function
        self.clear_grad()
        tr_loss = self.model.loss_fn(x_tr,y_tr)
        #print("training_leave_loss {}".format(tr_loss))
        tr_grad = self.get_gradients(tr_loss)

        #Get derivative w.r.t loss of test variable
        self.clear_grad()
        test_loss = self.model.loss_fn(x_test,y_test)
        #print("test_leave_loss {}".format(test_loss))
        test_grad = self.get_gradients(test_loss)
        
        #Get Hessian w.r.t whole training data
        self.clear_grad()
        loss = self.model.loss_fn(X_train, Y_train)
        #print("all_loss {}".format(loss))
        train_grad = self.get_gradients(loss)
        
        return tr_grad, test_grad, train_grad, tr_loss, test_loss, loss

    def get_influence( self, x_test, y_test, x_train_rem, y_train_rem, X_train, Y_train, print_output = False,
                         max_iterations = 100, load_hvp = -1 ):
        train_grad, test_grad, train_grad_all, tr_loss, test_loss, loss = influence_up(model, x_test, y_test, 
            x_train_rem, y_train_rem, X_train, Y_train)
    	
        if( load_hvp > 0 ):
            my_file = Path("./hvp_" + str(load_hvp) + ".npz" )
            if my_file.is_file():
                hessian_product = np.load(my_file)["h"].item()
            else:
                print("Computing again")
                hessian_product = self.get_hessian_product(train_grad_all, test_grad, print_output = print_output, 
                max_iterations = max_iterations )
                print(hessian_product)
                np.savez(my_file, h= hessian_product)
        else:
            hessian_product = self.get_hessian_product(train_grad_all, test_grad, print_output = print_output, max_iterations = max_iterations )

        influence = 0

        influence = influence + torch.sum(hessian_product * train_grad)
        print("influence {}".format(influence))

        return influence, hessian_product, train_grad, test_grad, train_grad_all, tr_loss, test_loss, loss



