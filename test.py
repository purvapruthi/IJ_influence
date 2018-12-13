from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torch.autograd as autograd
import numpy as np
from logReg import LogitReg
from hessian import *
from torch.nn import Parameter
import unittest



class MyTest(unittest.TestCase):
    def __init__(self, D_in = 10, D_out = 3, N = 10):
        super(MyTest, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.cls = LogitReg(2, D_in, D_out)
        self.cls.fc1.weight = Parameter(torch.randn(D_out,D_in),requires_grad = True)
        self.cls.fc1.bias = Parameter(torch.randn(D_out), requires_grad = True)
        self.w = self.cls.fc1.weight
        self.b = self.cls.fc1.bias
        self.N = N
        self.x = torch.randn(N,D_in)
        k = torch.randint(D_out-1, (N,))
        self.y = k.type(torch.LongTensor)
        self.vector = torch.randn(D_out,D_in)
    
    def forward(self, x):
        A = x @torch.t(self.w) + self.b
        m = nn.LogSoftmax(dim=1)
        forward = m(A)        
        return forward
    
    def double_gradient(self, x, y):
        N = x.shape[0]
        M = x.shape[1]
        loss = self.loss_fn(self.x,self.y)
        
        #loss.backward()
        
        #grad =  torch.autograd.grad( loss, self.w, create_graph=True)[0]
        #gradients = get_gradients(self.cls, cls_loss) 
        grad = torch.zeros(self.D_out, self.D_in, requires_grad = True)
        P = x @ torch.t(self.w) + self.b
        m = nn.Softmax(dim=1)
        prob = m(P)
        print(prob.shape)
        for i in range(N):
            for j in range(self.D_out):
                if j == y[i]:
                     grad[j,:] = grad[j,:] + (1 - prob[i,j]) * x[i]
                else:
                    grad[j,:] = grad[j,:] - prob[i,j] * x[i]
        
        grad = grad/N
        
        
        g = grad.contiguous().view(-1)
        l = g.size(0)
        double_gradient = torch.zeros(l,l)
        for idx in range(l):
            grad2rd = torch.autograd.grad( g[idx], self.w, create_graph=True)
            cnt = 0
            for elem in grad2rd:
                g2 = elem.contiguous().view(-1) if cnt == 0 else torch.cat([g2, elem.contiguous().view(-1)])
                cnt = 1
            double_gradient[idx] = g2
            
        print(np.linalg.eigvals(double_gradient.data.numpy()))
    
        return double_gradient
    
    def is_invertible(self, a):
        return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]
    
    def clear_grads(self):
        if self.w.grad is not None:
            self.w.grad.data.zero_()
            
        if self.b.grad is not None:
            self.b.grad.data.zero_()
        
    def split_data(self, tr_index, test_index):
        x_train = self.x[tr_index,: ].reshape(1,-1)
        y_train = self.y[tr_index].reshape(1,)
        
        print(x_train.shape)
        
        x_test = self.x[test_index,: ].reshape(1,-1)
        y_test = self.y[test_index].reshape(1,)
        
        print(x_test.shape)
        return x_train, y_train, x_test, y_test, self.x, self.y
    
    def loss_fn(self, x, y):
        fw = self.forward(x)
        y = y.type(torch.LongTensor)
        loss = F.nll_loss(fw,y)
        return loss
    
    def get_influence_t(self, tr_index, test_index):
        x_train, y_train, x_test, y_test, x, y = self.split_data( tr_index, test_index)
        
        self.clear_grads()
        tr_loss = self.loss_fn(x_train, y_train)
        print("training_leave_loss {}".format(tr_loss))
        tr_grad_w = torch.autograd.grad(tr_loss, self.w, retain_graph=True, create_graph=True)[0]
        tr_grad_b = torch.autograd.grad(tr_loss, self.b, retain_graph=True, create_graph=True)[0]
        
        self.clear_grads()
        test_loss = self.loss_fn(x_test, y_test)
        print("test_leave_loss {}".format(test_loss))
        test_grad_w = torch.autograd.grad(test_loss, self.w, retain_graph=True, create_graph=True)[0]
        test_grad_b = torch.autograd.grad(test_loss, self.b, retain_graph=True, create_graph=True)[0]
        
        
        self.clear_grads()
        train_loss = self.loss_fn(x, y)
        print("all_loss {}".format(train_loss))
        train_grad_w = torch.autograd.grad(train_loss, self.w, retain_graph=True, create_graph=True)[0]
        train_grad_b = torch.autograd.grad(train_loss, self.b, retain_graph=True, create_graph=True)[0]
        
        return tr_grad_w, tr_grad_b, test_grad_w, test_grad_b, train_grad_w, train_grad_b
        
    def test_forward(self):
        fw = self.forward(self.x)
        self.assertEqual(torch.equal(self.cls.forward(self.x), fw), True )
    
    def test_loss(self):
        loss = self.loss_fn(self.x, self.y)
        self.assertEqual(torch.equal(self.cls.loss_fn(self.x,self.y), loss), True )
        
    def test_clear_gradients(self):
        loss = self.cls.loss_fn(self.x, self.y)
        loss.backward()
        
        self.assertEqual(torch.equal(self.cls.fc1.weight.grad, torch.zeros_like(self.cls.fc1.weight.grad)), False)
        self.assertEqual(torch.equal(self.cls.fc1.bias.grad, torch.zeros_like(self.cls.fc1.bias.grad)), False)
        clear_grad(self.cls)
        
        self.assertEqual(torch.equal(self.cls.fc1.weight.grad, torch.zeros_like(self.cls.fc1.weight.grad)), True)
        self.assertEqual(torch.equal(self.cls.fc1.bias.grad, torch.zeros_like(self.cls.fc1.bias.grad)), True)
     
    def test_get_gradients(self ):
        loss = self.loss_fn(self.x, self.y)
        cls_loss = self.cls.loss_fn(self.x,self.y)
        loss.backward()
        gradients = get_gradients(self.cls, cls_loss) 
        self.assertEqual(torch.equal(gradients['fc1.weight'], self.w.grad ), True)
        self.assertEqual(torch.equal(gradients['fc1.bias'], self.b.grad ), True)
        
    def test_hvp(self):
        loss = self.loss_fn(self.x, self.y)
        grad = torch.autograd.grad(loss, self.w, retain_graph=True, create_graph=True)[0]
        
        hvp = hvp_computation(grad, self.vector, self.w )
        g = grad.contiguous().view(-1)
        v = self.vector.contiguous().view(-1)
        hvp_cont = hvp.contiguous().view(-1)
    
        hessian = eval_hessian(g, self.w)
        actual_hvp = hessian @ v
        self.assertAlmostEqual( torch.isclose(hvp_cont,actual_hvp).all(), True) 
        
    def test_hessian_ivp_1(self, N):
        A = torch.rand(N,N)
        x = torch.randn(N,1)
        x.requires_grad = True
        
        B = torch.randn(N,1)
        
        H = A @ torch.t(A)
        print(np.linalg.eigvals(H.data.numpy()) > 0)
        func = (1/2) * torch.t(x) @ H @ x - torch.t(B) @ x
        
        grad1 = autograd.grad(func, x, create_graph=True, retain_graph= True)[0]
        grad2 = eval_hessian(grad1, x)
        solution = torch.inverse(grad2) @ B
        exp_solution = conjugate_gradient_optimization( grad1, B, x, max_iterations = 10000, print_output = False)
        #self.assertEqual( torch.isclose(torch.norm(solution),torch.norm(exp_solution), 10), True)
        return torch.norm(solution), torch.norm(exp_solution), torch.norm(H@solution - B), torch.norm(H@exp_solution - B)
        
    def test_grad_leave_out(self, tr_index, test_index):
        
        x_train, y_train, x_test, y_test, x, y = self.split_data( tr_index, test_index)
        tr_grad_w, tr_grad_b, test_grad_w, test_grad_b, train_grad_w, train_grad_b = self.get_influence_t(tr_index, test_index)
        
        actual_tr_grad, actual_test_grad, actual_train_grad = influence_up(self.cls, x_test, y_test, x_train, y_train, x, y)
        self.assertEqual( torch.isclose(torch.norm(actual_tr_grad['fc1.weight']),torch.norm(tr_grad_w)), True)
        self.assertEqual( torch.isclose(torch.norm(actual_test_grad['fc1.weight']),torch.norm(test_grad_w)), True)
        self.assertEqual( torch.isclose(torch.norm(actual_train_grad['fc1.weight']),torch.norm(train_grad_w)), True)
        self.assertEqual( torch.isclose(torch.norm(actual_tr_grad['fc1.bias']),torch.norm(tr_grad_b)), True)
        self.assertEqual( torch.isclose(torch.norm(actual_test_grad['fc1.bias']),torch.norm(test_grad_b) ), True)
        self.assertEqual( torch.isclose(torch.norm(actual_train_grad['fc1.bias']),torch.norm(train_grad_b) ), True)
    
    def exact_solution(self, H, b):
        H_np = H.data.detach().numpy()
        b_np = b.data.detach().numpy()
        solution = np.linalg.solve(H_np, b_np)
        
        return torch.from_numpy(solution)
        
    def test_getting_influence(self, tr_index, test_index):
        x_train, y_train, x_test, y_test, x, y = self.split_data( tr_index, test_index)
        
        self.clear_grads()
        tr_loss = self.loss_fn(x_train, y_train)
        print("training_leave_loss {}".format(tr_loss))
        tr_grad_w = torch.autograd.grad(tr_loss, self.w, create_graph=True)[0]
        tr_grad_b = torch.autograd.grad(tr_loss, self.b, create_graph=True)[0]
        
        self.clear_grads()
        test_loss = self.loss_fn(x_test, y_test)
        print("test_leave_loss {}".format(test_loss))
        test_grad_w = torch.autograd.grad(test_loss, self.w, create_graph=True)[0]
        test_grad_b = torch.autograd.grad(test_loss, self.b, create_graph=True)[0]
        
        
        self.clear_grads()
        train_loss = self.loss_fn(x, y)
        print("all_loss {}".format(train_loss))
        train_grad_w = torch.autograd.grad(train_loss, self.w, retain_graph=True, create_graph=True)[0]
        train_grad_b = torch.autograd.grad(train_loss, self.b, retain_graph= True, create_graph=True)[0]
        
        grad = train_grad_w.contiguous().view(-1)
        b = test_grad_w.contiguous().view(-1)
       
        hessian = eval_hessian(grad, self.w)
        hk = hessian.data.numpy()
        
        print(np.linalg.eigvals(hk) > 0)
        print("Is invertible {}".format(self.is_invertible(hk)))
        
        sol_linalg = self.exact_solution(hessian,b)
        print( "Exact Residual {}".format(hessian @ sol_linalg - b))
 
        sol = conjugate_gradient_optimization( train_grad_w, test_grad_w, self.w, print_output=True)
        residual = hessian @ sol.view(-1) - b
        print( "Residual {}".format(residual))
        print(sol.view(-1))
        print(sol_linalg)

        self.assertEqual(torch.isclose(torch.norm(residual), torch.zeros(residual.shape), atol=1e-4).all(), True )