import pdb
import torch
import numpy as np

from mytorch.tensor import Tensor
from mytorch.autograd_engine import *
from mytorch.nn.functional import *
from mytorch.nn.sequential import Sequential
from autograder.hw1_autograder.test_mlp import *
from autograder.helpers import *
from mytorch.nn.linear import *
from mytorch.nn.conv import *


import os
import sys


import traceback

import matplotlib.pyplot as plt
import numpy as np

from hw1.mnist import mnist

"""Use this file to help you develop operations/functions.
It actually works fairly similarly to the autograder.
We've provided many test functions.
For your own operations, implement tests for them here to easily
debug your code."""

def main():
    """Runs test methods in order shown below."""
    test_flatten()
    #test_conv1d_backward()
    '''
    # test four basic ops
    #pdb.set_trace()
    test_add()
    test_sub()
    #pdb.set_trace()
    test_mul()
    test_div()

    # you probably want to verify
    # any other ops you create...
    test_matmul()
    test_linear_forward()
    test_linear_backward()

    test_linear_relu_forward()
    test_linear_relu_backward()
    test_big_linear_relu_forward()
    test_big_linear_relu_backward()

    test_linear_relu_step()
    test_big_linear_relu_step()

    test_linear_xeloss_forward()
    test_linear_xeloss_backward()
    test_big_linear_bn_relu_xeloss_train_eval() 
    test_big_linear_relu_xeloss_train_eval()


    test_linear_momentum()
    test_big_linear_batchnorm_relu_xeloss_momentum()
    test_big_linear_relu_xeloss_momentum()


    test_linear_batchnorm_relu_forward_train()
    test_linear_batchnorm_relu_backward_train()
    test_linear_batchnorm_relu_train_eval()
    test_big_linear_batchnorm_relu_train_eval()



    # Test and print MLP for MNIST
    #train_x, train_y, val_x, val_y = load_data()
    #val_accuracies = mnist(train_x, train_y, val_x, val_y)
    #visualize_results(val_accuracies)


    # test autograd
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    test7()
    test8()

    # for when you might want it...
    testbroadcast()
    '''

# Linear and relu forward/backward tests
def test_linear_forward():
    np.random.seed(11785)
    mytorch_mlp = Sequential(Linear(10, 20))
    test_forward(mytorch_mlp)
    check_model_param_settings(mytorch_mlp)
    return True
def test_linear_backward():
    np.random.seed(11785)
    mytorch_mlp = Sequential(Linear(10, 20))
    test_forward_backward(mytorch_mlp)
    return True


#Linear-relu
def test_linear_relu_forward():
    np.random.seed(11785)
    mytorch_mlp = Sequential(Linear(10, 20), ReLU())
    test_forward(mytorch_mlp)
    return True
def test_linear_relu_backward():
    np.random.seed(11785)
    mytorch_mlp = Sequential(Linear(10, 20), ReLU())
    test_forward_backward(mytorch_mlp)
    return True
def test_big_linear_relu_forward():
    np.random.seed(11785)
    mytorch_mlp = Sequential(Linear(10, 20), ReLU(), Linear(20, 30), ReLU())
    test_forward(mytorch_mlp)
    return True
def test_big_linear_relu_backward():
    np.random.seed(11785)
    mytorch_mlp = Sequential(Linear(10, 20), ReLU(), Linear(20, 30), ReLU())
    test_forward_backward(mytorch_mlp)
    return True


#SGD 
def test_linear_relu_step():
    np.random.seed(11785)
    mytorch_mlp = Sequential(Linear(10, 20), ReLU())
    mytorch_optimizer = SGD(mytorch_mlp.parameters())
    test_step(mytorch_mlp, mytorch_optimizer, 5, 5)
    return True
def test_big_linear_relu_step():
    np.random.seed(11785)
    mytorch_mlp = Sequential(Linear(10, 20),  ReLU(), Linear(20, 30), ReLU())
    mytorch_optimizer = SGD(mytorch_mlp.parameters())
    test_step(mytorch_mlp, mytorch_optimizer, 5, 5)
    return True



# cross entropy tests
def test_linear_xeloss_forward():
    np.random.seed(11785)
    mytorch_mlp = Sequential(Linear(10, 20))
    mytorch_optimizer = SGD(mytorch_mlp.parameters())
    mytorch_criterion = CrossEntropyLoss()
    test_forward(mytorch_mlp, mytorch_criterion=mytorch_criterion)
    return True


def test_linear_xeloss_backward():
    np.random.seed(11785)
    mytorch_mlp = Sequential(Linear(10, 20))
    mytorch_optimizer = SGD(mytorch_mlp.parameters())
    mytorch_criterion = CrossEntropyLoss()
    test_forward_backward(mytorch_mlp, mytorch_criterion=mytorch_criterion)
    return True


def test_big_linear_bn_relu_xeloss_train_eval():
    np.random.seed(11785)
    mytorch_mlp = Sequential(Linear(10, 20), BatchNorm1d(20), ReLU(), Linear(20, 30), BatchNorm1d(30), ReLU())
    mytorch_optimizer = SGD(mytorch_mlp.parameters())
    mytorch_criterion = CrossEntropyLoss()
    test_step(mytorch_mlp, mytorch_optimizer, 5, 5, mytorch_criterion=mytorch_criterion)
    return True

def test_big_linear_relu_xeloss_train_eval():
    np.random.seed(11785)
    mytorch_mlp = Sequential(Linear(10, 20), ReLU(), Linear(20, 30), ReLU())
    mytorch_optimizer = SGD(mytorch_mlp.parameters())
    mytorch_criterion = CrossEntropyLoss()
    test_step(mytorch_mlp, mytorch_optimizer, 5, 5, mytorch_criterion=mytorch_criterion)
    return True



# momentum tests
def test_linear_momentum():
    np.random.seed(11785)
    mytorch_mlp = Sequential(Linear(10, 20), ReLU())
    mytorch_optimizer = SGD(mytorch_mlp.parameters(), momentum=0.9)
    test_step(mytorch_mlp, mytorch_optimizer, 5, 0)
    return True


def test_big_linear_batchnorm_relu_xeloss_momentum():
    np.random.seed(11785)
    mytorch_mlp = Sequential(Linear(10, 20), BatchNorm1d(20), ReLU(),
                             Linear(20, 30), BatchNorm1d(30), ReLU())
    mytorch_optimizer = SGD(mytorch_mlp.parameters(), momentum = 0.9)
    mytorch_criterion = CrossEntropyLoss()
    test_step(mytorch_mlp, mytorch_optimizer, 5, 5, mytorch_criterion = mytorch_criterion)
    return True

def test_big_linear_relu_xeloss_momentum():
    np.random.seed(11785)
    mytorch_mlp = Sequential(Linear(10, 20), ReLU(),
                             Linear(20, 30), ReLU())
    mytorch_optimizer = SGD(mytorch_mlp.parameters(), momentum = 0.9)
    mytorch_criterion = CrossEntropyLoss()
    test_step(mytorch_mlp, mytorch_optimizer, 5, 5, mytorch_criterion = mytorch_criterion)
    return True



# batchnorm tests
def test_linear_batchnorm_relu_forward_train():
    np.random.seed(11785)
    mytorch_mlp = Sequential(Linear(10, 20), BatchNorm1d(20), ReLU())
    test_forward(mytorch_mlp)
    return True


def test_linear_batchnorm_relu_backward_train():
    np.random.seed(11785)
    mytorch_mlp = Sequential(Linear(10, 20), BatchNorm1d(20), ReLU())
    test_forward_backward(mytorch_mlp)
    return True


def test_linear_batchnorm_relu_train_eval():
    np.random.seed(11785)
    mytorch_mlp = Sequential(Linear(10, 20), BatchNorm1d(20), ReLU())
    mytorch_optimizer = SGD(mytorch_mlp.parameters())
    test_step(mytorch_mlp, mytorch_optimizer, 5, 5)
    return True


def test_big_linear_batchnorm_relu_train_eval():
    np.random.seed(11785)
    mytorch_mlp = Sequential(Linear(10, 20), BatchNorm1d(20), ReLU())
    mytorch_optimizer = SGD(mytorch_mlp.parameters())
    test_step(mytorch_mlp, mytorch_optimizer, 5, 5)
    return True



def test_add():
    """Tests that mytorch addition matches torch's addition"""

    # shape of tensor to test
    shape = (1, 2, 3)

    # get mytorch and torch tensor: 'a'
    a = Tensor.randn(*shape)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    # get mytorch and torch tensor: 'b'
    b = Tensor.randn(*shape)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)

    # run mytorch and torch forward: 'c = a + b'
    ctx = ContextManager()
    c = Add.forward(ctx, a, b)
    c_torch = a_torch + b_torch

    # run mytorch and torch addition backward
    back = Add.backward(ctx, Tensor.ones(*shape))
    c_torch.sum().backward()

    # check that c matches
    assert check_val_and_grad(c, c_torch)
    # check that dc/da and dc/db respectively match
    assert check_val_and_grad(back[0], a_torch.grad)
    assert check_val_and_grad(back[1], b_torch.grad)

    # ensure + is overridden
    c_using_override = a + b
    assert check_val(c_using_override, c_torch)

    return True

def test_sub():
    """Tests that mytorch subtraction matches torch's subtraction"""

    # shape of tensor to test
    shape = (1, 2, 3)

    # get mytorch and torch tensor: 'a'
    a = Tensor.randn(*shape)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    # get mytorch and torch tensor: 'b'
    b = Tensor.randn(*shape)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)

    # run mytorch and torch forward: 'c = a - b'
    ctx = ContextManager()
    c = Sub.forward(ctx, a, b)
    c_torch = a_torch - b_torch

    # run mytorch and torch subtraction backward
    back = Sub.backward(ctx, Tensor.ones(*shape))
    c_torch.sum().backward()

    # check that c matches
    assert check_val_and_grad(c, c_torch)
    # check that dc/da and dc/db respectively match
    assert check_val(back[0], a_torch.grad)
    assert check_val(back[1], b_torch.grad)

    # ensure - is overridden
    c_using_override = a - b
    assert check_val(c_using_override, c_torch)

    return True

def test_mul():
    """Tests that mytorch's elementwise multiplication matches torch's"""

    # shape of tensor to test
    shape = (1, 2, 3)

    # get mytorch and torch tensor: 'a'
    a = Tensor.randn(*shape)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    # get mytorch and torch tensor: 'b'
    b = Tensor.randn(*shape)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)

    # run mytorch and torch forward: 'c = a * b'
    ctx = ContextManager()
    c = Mul.forward(ctx, a, b)
    c_torch = a_torch * b_torch
    # run mytorch and torch multiplication backward
    back = Mul.backward(ctx, Tensor.ones(*shape))
    c_torch.sum().backward()

    # check that c matches
    assert check_val_and_grad(c, c_torch)
    # check that dc/da and dc/db respectively match
    assert check_val(back[0], a_torch.grad)
    assert check_val(back[1], b_torch.grad)

    # ensure * is overridden
    c_using_override = a * b
    assert check_val(c_using_override, c_torch)

    return True

def test_matmul():
    """Tests that mytorch's  matmul multiplication matches torch's"""

    # shape of tensor to test
    shape1 = (3,4)
    shape2 = (4,5)
    shape = (3,5)

    # get mytorch and torch tensor: 'a'
    a = Tensor.randn(*shape1)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    # get mytorch and torch tensor: 'b'
    b = Tensor.randn(*shape2)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)

    # run mytorch and torch forward: 'c = a * b'
    ctx = ContextManager()
    c = matmul.forward(ctx, a, b)
    c_torch = torch.matmul(a_torch,b_torch)
    # run mytorch and torch multiplication backward
    back = matmul.backward(ctx, Tensor.ones(*shape))
    c_torch.sum().backward()

    # check that c matches
    assert check_val_and_grad(c, c_torch)
    # check that dc/da and dc/db respectively match
    assert check_val(back[0], a_torch.grad)
    assert check_val(back[1], b_torch.grad)

    return True

def test_div():
    """Tests that mytorch division matches torch's"""

    # shape of tensor to test
    shape = (1, 2, 3)

    # get mytorch and torch tensor: 'a'
    a = Tensor.randn(*shape)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    # get mytorch and torch tensor: 'b'
    b = Tensor.randn(*shape)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)

    # run mytorch and torch forward: 'c = a / b'
    ctx = ContextManager()
    c = Div.forward(ctx, a, b)
    c_torch = a_torch / b_torch

    # run mytorch and torch division backward
    back = Div.backward(ctx, Tensor.ones(*shape))
    c_torch.sum().backward()

    # check that c matches
    assert check_val_and_grad(c, c_torch)
    # check that dc/da and dc/db respectively match
    assert check_val(back[0], a_torch.grad)
    assert check_val(back[1], b_torch.grad)

    # ensure / is overridden
    c_using_override = a / b
    assert check_val(c_using_override, c_torch)

    return True


def testbroadcast():
    """Tests addition WITH broadcasting matches torch's"""

    # shape of tensor to test

    # get mytorch and torch tensor: 'a'
    a = Tensor.randn(3, 4)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    # get mytorch and torch tensor: 'b'
    b = Tensor.randn(4)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)

    # run mytorch and torch forward: 'c = a + b'
    c = a + b
    c_torch = a_torch + b_torch

    # run mytorch and torch addition backward
    c.backward()
    c_torch.sum().backward()

    # check that c matches
    assert check_val_and_grad(c, c_torch)
    # check that dc/da and dc/db respectively match
    assert check_val_and_grad(a, a_torch)
    assert check_val_and_grad(b, b_torch)


# addition, requires grad
def test1():
    a = Tensor.randn(1, 2, 3)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    b = Tensor.randn(1, 2, 3)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)
    

    c = a + b
    c_torch = a_torch + b_torch

    c_torch.sum().backward()
    #pdb.set_trace()
    c.backward()

    assert check_val_and_grad(a, a_torch)
    assert check_val_and_grad(b, b_torch)
    assert check_val_and_grad(c, c_torch)
    

# multiplication, requires grad
def test2():
    a = Tensor.randn(1, 2, 3)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    b = Tensor.randn(1, 2, 3)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)

    c = a * b
    c_torch = a_torch * b_torch

    c_torch.sum().backward()
    c.backward()

    assert check_val_and_grad(a, a_torch)
    assert check_val_and_grad(b, b_torch)
    assert check_val_and_grad(c, c_torch)

# addition, one arg requires grad
def test3():
    a = Tensor.randn(1, 2, 3)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    b = Tensor.randn(1, 2, 3)
    b.requires_grad = False
    b_torch = get_same_torch_tensor(b)

    c = a + b
    c_torch = a_torch + b_torch

    c_torch.sum().backward()
    c.backward()

    assert check_val_and_grad(a, a_torch)
    assert check_val_and_grad(b, b_torch)
    assert check_val_and_grad(c, c_torch)

# the example from writeup
def test4():
    a = Tensor(1, requires_grad = True)
    a_torch = get_same_torch_tensor(a)

    b = Tensor(2, requires_grad = True)
    b_torch = get_same_torch_tensor(b)

    c = Tensor(3, requires_grad = True)
    c_torch = get_same_torch_tensor(c)

    #pdb.set_trace()
    d = a + a * b
    d_torch = a_torch + a_torch * b_torch

    e = d + c + Tensor(3)
    e_torch = d_torch + c_torch + torch.tensor(3)

    e.backward()
    e_torch.sum().backward()

    assert check_val_and_grad(a, a_torch)
    assert check_val_and_grad(b, b_torch)
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(d, d_torch)
    assert check_val_and_grad(e, e_torch)


# the example from writeup, more strict
def test5():
    a = Tensor(1, requires_grad = True)
    a_torch = get_same_torch_tensor(a)

    b = Tensor(2, requires_grad = True)
    b_torch = get_same_torch_tensor(b)

    c = Tensor(3, requires_grad = True)
    c_torch = get_same_torch_tensor(c)

    # d = a + a * b
    z1 = a * b
    z1_torch = a_torch * b_torch
    d = a + z1
    d_torch = a_torch + z1_torch

    # e = (d + c) + 3
    z2 = d + c
    z2_torch = d_torch + c_torch
    e = z2 + Tensor(3)
    e_torch = z2_torch + 3

    e.backward()
    e_torch.sum().backward()

    assert check_val_and_grad(a, a_torch)
    assert check_val_and_grad(b, b_torch)
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(z1, z1_torch)
    assert check_val_and_grad(d, d_torch)
    assert check_val_and_grad(z2, z2_torch)
    assert check_val_and_grad(e, e_torch)


# more complicated tests
def test6():
    a = Tensor.randn(2, 3)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    b = Tensor.randn(2, 3)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)

    c = a / b
    c_torch = a_torch / b_torch

    d = a - b
    d_torch = a_torch - b_torch

    e = c + d
    e_torch = c_torch + d_torch

    e.backward()
    e_torch.sum().backward()

    assert check_val_and_grad(a, a_torch)
    assert check_val_and_grad(b, b_torch)
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(d, d_torch)
    assert check_val_and_grad(e, e_torch)


# another fun test
def test7():
    # a = 3
    a = Tensor(3., requires_grad=False)
    a_torch = get_same_torch_tensor(a)

    # b = 4
    b = Tensor(4., requires_grad=False)
    b_torch = get_same_torch_tensor(b)

    # c = 5
    c = Tensor(5., requires_grad=True)
    c_torch = get_same_torch_tensor(c)

    # out = a * b + 3 * c
    z1 = a * b
    z1_torch = a_torch * b_torch
    z2 = Tensor(3) * c
    z2_torch = 3 * c_torch
    out = z1 + z2
    out_torch = z1_torch + z2_torch

    out_torch.sum().backward()
    out.backward()

    assert check_val_and_grad(a, a_torch)
    assert check_val_and_grad(b, b_torch)
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(z1, z1_torch)
    assert check_val_and_grad(z2, z2_torch)
    assert check_val_and_grad(out, out_torch)

# non-tensor arguments
def test8():
    a = Tensor.randn(1, 2, 3)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    b = Tensor.randn(1, 2, 3)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)

    c = a + b
    c_torch = a_torch + b_torch

    d = c.reshape(-1)
    d_torch = c_torch.reshape(-1)

    d_torch.sum().backward()
    d.backward()

    assert check_val_and_grad(a, a_torch)
    assert check_val_and_grad(b, b_torch)
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(d, d_torch)


"""General-use helper functions"""

def get_same_torch_tensor(mytorch_tensor):
    """Returns a torch tensor with the same data/params as some mytorch tensor"""
    res = torch.tensor(mytorch_tensor.data).double()
    res.requires_grad = mytorch_tensor.requires_grad
    return res


def check_val_and_grad(mytorch_tensor, pytorch_tensor):
    """Compares values and params of mytorch and torch tensors.
    
    Returns:
        boolean: False if not similar, True if similar"""
    return check_val(mytorch_tensor, pytorch_tensor) and \
           check_grad(mytorch_tensor, pytorch_tensor)


def check_val(mytorch_tensor, pytorch_tensor, eps=1e-10):
    """Compares the data values of mytorch/torch tensors."""
    if not isinstance(pytorch_tensor, torch.DoubleTensor):
        print("Warning: torch tensor is not a DoubleTensor. It is instead {}".format(pytorch_tensor.type()))
        print("It is highly recommended that similarity testing is done with DoubleTensors as numpy arrays have 64-bit precision (like DoubleTensors)")

    if tuple(mytorch_tensor.shape) != tuple(pytorch_tensor.shape):
        print("mytorch tensor and pytorch tensor has different shapes: {}, {}".format(
            mytorch_tensor.shape, pytorch_tensor.shape
        ))
        return False
    data_diff = np.abs(mytorch_tensor.data - pytorch_tensor.data.numpy())
    max_diff = data_diff.max()
    if max_diff < eps:
        return True
    else:
        print("Data element differs by {}:".format(max_diff))
        print("mytorch tensor:")
        print(mytorch_tensor)
        print("pytorch tensor:")
        print(pytorch_tensor)

        return False

def check_grad(mytorch_tensor, pytorch_tensor, eps = 1e-10):
    """Compares the gradient of mytorch and torch tensors"""
    if mytorch_tensor.grad is None or pytorch_tensor_nograd(pytorch_tensor):
        if mytorch_tensor.grad is None and pytorch_tensor_nograd(pytorch_tensor):
            return True
        elif mytorch_tensor.grad is None:
            print("Mytorch grad is None, but pytorch is not")
            return False
        else:
            print("Pytorch grad is None, but mytorch is not")
            return False

    grad_diff = np.abs(mytorch_tensor.grad.data - pytorch_tensor.grad.data.numpy())
    max_diff = grad_diff.max()
    if max_diff < eps:
        return True
    else:
        print("Grad differs by {}".format(grad_diff))
        return False

def pytorch_tensor_nograd(pytorch_tensor):
    return not pytorch_tensor.requires_grad or not pytorch_tensor.is_leaf


def conv1d_forward_correctness(num_layers=1):
    '''
    CNN: scanning with a MLP with stride
    '''
    scores_dict = [0]

    ############################################################################################
    #############################   Initialize parameters    ###################################
    ############################################################################################
    in_c = np.random.randint(5, 15)
    channels = [np.random.randint(5, 15) for i in range(num_layers + 1)]
    kernel = [np.random.randint(3,7) for i in range(num_layers)]
    stride = [np.random.randint(3,5) for i in range(num_layers)]
    width = np.random.randint(60,80)
    batch_size = np.random.randint(1,4)

    x = np.random.randn(batch_size, channels[0], width)

    #############################################################################################
    #################################    Create Models   ########################################
    #############################################################################################
    test_layers = [Conv1d(channels[i], channels[i + 1], kernel[i], stride[i])
                    for i in range(num_layers)]
    test_model = Sequential(*test_layers)


    torch_layers = [nn.Conv1d(channels[i], channels[i + 1], kernel[i], stride=stride[i])
                    for i in range(num_layers)]
    torch_model = nn.Sequential(*torch_layers)

    for torch_layer, test_layer in zip(torch_model, test_model.layers):
        torch_layer.weight = nn.Parameter(torch.tensor(test_layer.weight.data))
        torch_layer.bias = nn.Parameter(torch.tensor(test_layer.bias.data))
    #############################################################################################
    #########################    Get the correct results from PyTorch   #########################
    #############################################################################################
    x1 = Variable(torch.tensor(x), requires_grad=True)
    y1 = torch_model(x1)
    torch_y = y1.detach().numpy()

    #############################################################################################
    ###################    Get fwd results from TestModel and compare  ##########################
    #############################################################################################
    y2 = test_model(Tensor(x))
    test_y = y2.data

    # check that model is correctly configured
    check_model_param_settings(test_model)

    if not assertions(test_y, torch_y, 'type', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'shape', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'closeness', 'y'): return scores_dict
    scores_dict[0] = 1

    return scores_dict

def test_conv1d_forward():
    np.random.seed(11785)
    n = 3
    for i in range(n):
        a = conv1d_forward_correctness(i+1)[0]
        if a != 1:
            print('Failed Conv1D Forward Test: %d / %d' % (i + 1, n))
            return False
        else:
            print('Passed Conv1D Forward Test: %d / %d' % (i + 1, n))
    return True

def conv1d_backward_correctness(num_layers = 1):
    '''
    CNN: scanning with a MLP with stride
    '''
    scores_dict = [0, 0, 0, 0]

    ############################################################################################
    #############################   Initialize parameters    ###################################
    ############################################################################################
    in_c = np.random.randint(5, 15)
    channels = [np.random.randint(5, 15) for i in range(num_layers + 1)]
    kernel = [np.random.randint(3,7) for i in range(num_layers)]
    stride = [np.random.randint(3,5) for i in range(num_layers)]
    width = np.random.randint(60,80)
    batch_size = np.random.randint(1,4)

    x = np.random.randn(batch_size, channels[0], width)

    #############################################################################################
    #################################    Create Models   ########################################
    #############################################################################################
    test_layers = [Conv1d(channels[i], channels[i + 1], kernel[i], stride[i])
                    for i in range(num_layers)]
    test_model = Sequential(*test_layers)


    torch_layers = [nn.Conv1d(channels[i], channels[i + 1], kernel[i], stride=stride[i])
                    for i in range(num_layers)]
    torch_model = nn.Sequential(*torch_layers)

    for torch_layer, test_layer in zip(torch_model, test_model.layers):
        torch_layer.weight = nn.Parameter(torch.tensor(test_layer.weight.data))
        torch_layer.bias = nn.Parameter(torch.tensor(test_layer.bias.data))

    #############################################################################################
    #########################    Get the correct results from PyTorch   #########################
    #############################################################################################
    x1 = Variable(torch.tensor(x), requires_grad=True)
    y1 = torch_model(x1)
    torch_y = y1.detach().numpy()

    b, c, w = torch_y.shape
    y1.sum().backward()
    dx1 = x1.grad
    torch_dx = dx1.detach().numpy()

    #############################################################################################
    ###################    Get fwd results from TestModel and compare  ##########################
    #############################################################################################
    x_tensor = Tensor(x, requires_grad=True)
    y2 = test_model(x_tensor)
    test_y = y2.data

    # check that model is correctly configured
    check_model_param_settings(test_model)

    if not assertions(test_y, torch_y, 'type', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'shape', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'closeness', 'y'): return scores_dict
    scores_dict[0] = 1

    #############################################################################################
    ###################    Get bwd results from TestModel and compare  ##########################
    #############################################################################################
    y2.backward()
    test_dx = x_tensor.grad.data

    # check that model is correctly configured
    check_model_param_settings(test_model)

    if not assertions(test_dx, torch_dx, 'type', 'dx'): return scores_dict
    if not assertions(test_dx, torch_dx, 'shape', 'dx'): return scores_dict
    if not assertions(test_dx, torch_dx, 'closeness', 'dx'): return scores_dict
    scores_dict[1] = 1


    for torch_layer, test_layer in zip(torch_model, test_model.layers):
        torch_dW = torch_layer.weight.grad.detach().numpy()
        torch_db = torch_layer.bias.grad.detach().numpy()
        test_dW = test_layer.weight.grad.data
        test_db = test_layer.bias.grad.data

        if not assertions(test_dW, torch_dW, 'type', 'dW'): return scores_dict
        if not assertions(test_dW, torch_dW, 'shape', 'dW'): return scores_dict
        if not assertions(test_dW, torch_dW, 'closeness', 'dW'): return scores_dict

        if not assertions(test_db, torch_db, 'type', 'db'): return scores_dict
        if not assertions(test_db, torch_db, 'shape', 'db'): return scores_dict
        if not assertions(test_db, torch_db, 'closeness', 'db'): return scores_dict

    scores_dict[2] = 1
    scores_dict[3] = 1

    check_model_param_settings(test_model)

    #############################################################################################
    ##############################    Compare Results   #########################################
    #############################################################################################

    return scores_dict

def test_conv1d_backward():
    np.random.seed(11785)
    n = 3
    for i in range(n):
        a, b, c, d = conv1d_backward_correctness(num_layers=i+1)
        if a != 1:
            print('Failed Conv1D Forward Test: %d / %d' % (i + 1, n))
            return False
        elif b != 1 or c != 1 or d != 1:
            print('Failed Conv1D Backward Test: %d / %d' % (i + 1, n))
            return False
        else:
            print('Passed Conv1D Backward Test: %d / %d' % (i + 1, n))
    return True


############################################################################################
##################################   Section  - Flatten   ################################
############################################################################################

def flatten_correctness():
    '''
    Flatten Layer
    '''
    scores_dict = [0, 0, 0, 0]

    ############################################################################################
    #############################   Initialize parameters    ###################################
    ############################################################################################
    in_c = np.random.randint(5,15)
    width = np.random.randint(60,80)
    batch_size = np.random.randint(1,4)

    x1d = np.random.randn(batch_size, in_c, width)
    x2d = np.random.randn(batch_size, in_c, width, width)
    x1d_tensor = Tensor(x1d, requires_grad=True)
    x2d_tensor = Tensor(x2d, requires_grad=True)

    #############################################################################################
    #################################    Create Models   ########################################
    #############################################################################################
    test_model = Flatten()

    torch_model = nn.Flatten()

    #############################################################################################
    #########################    Get the correct results from PyTorch   #########################
    #############################################################################################
    x1 = Variable(torch.tensor(x1d), requires_grad=True)
    y1 = torch_model(x1)
    torch_y = y1.detach().numpy()

    y1.sum().backward()
    dx1 = x1.grad
    torch_x = dx1.detach().numpy()

    #############################################################################################
    ###################    Get fwd results from TestModel and compare  ##########################
    #############################################################################################
    y2 = test_model(x1d_tensor)
    test_y = y2.data

    if not assertions(test_y, torch_y, 'type', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'shape', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'closeness', 'y'): return scores_dict
    scores_dict[0] = 1

    #############################################################################################
    ###################    Get bwd results from TestModel and compare  ##########################
    #############################################################################################
    y2.backward()
    dx2 = x1d_tensor.grad.data
    test_x = dx2

    if not assertions(test_x, torch_x, 'type', 'x'): return scores_dict
    if not assertions(test_x, torch_x, 'shape', 'x'): return scores_dict
    if not assertions(test_x, torch_x, 'closeness', 'x'): return scores_dict
    scores_dict[1] = 1

    #############################################################################################
    #########################    Get the correct results from PyTorch   #########################
    #############################################################################################
    x1 = Variable(torch.tensor(x2d), requires_grad=True)
    y1 = torch_model(x1)
    torch_y = y1.detach().numpy()

    y1.sum().backward()
    dx1 = x1.grad
    torch_x = dx1.detach().numpy()

    #############################################################################################
    ###################    Get fwd results from TestModel and compare  ##########################
    #############################################################################################
    y2 = test_model(x2d_tensor)
    test_y = y2.data

    if not assertions(test_y, torch_y, 'type', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'shape', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'closeness', 'y'): return scores_dict
    scores_dict[2] = 1

    #############################################################################################
    ###################    Get bwd results from TestModel and compare  ##########################
    #############################################################################################
    y2.backward()
    dx2 = x2d_tensor.grad.data
    test_x = dx2

    if not assertions(test_x, torch_x, 'type', 'x'): return scores_dict
    if not assertions(test_x, torch_x, 'shape', 'x'): return scores_dict
    if not assertions(test_x, torch_x, 'closeness', 'x'): return scores_dict
    scores_dict[3] = 1

    return scores_dict

def test_flatten():
    np.random.seed(11785)
    n = 3
    for i in range(n):
        a, b, c, d = flatten_correctness()
        if a != 1:
            print('Failed Flatten Forward Test: %d / %d' % (i + 1, n))
            return False
        elif b != 1:
            print('Failed Flatten Backward Test: %d / %d' % (i + 1, n))
            return False
        elif c != 1:
            print('Failed Flatten Forward Test: %d / %d' % (i + 1, n))
            return False
        elif d != 1:
            print('Failed Flatten Backward Test: %d / %d' % (i + 1, n))
            return False
        else:
            print('Passed Flatten Test: %d / %d' % (i + 1, n))
    return True

    


def load_data():
    DATA_PATH = './autograder/hw1_autograder/data'
    train_x = np.load(os.path.join(DATA_PATH, "train_data.npy"))
    train_y = np.load(os.path.join(DATA_PATH, "train_labels.npy"))
    val_x = np.load(os.path.join(DATA_PATH, "val_data.npy"))
    val_y = np.load(os.path.join(DATA_PATH, "val_labels.npy"))

    train_x = train_x / 255
    val_x = val_x / 255

    return train_x, train_y, val_x, val_y


def visualize_results(val_accuracies):
    print("Saving and showing graph")
    try:
        plt.plot(val_accuracies)
        plt.ylabel('Accuracy')
        plt.savefig('hw1/validation_accuracy.png')
        print("Accuracies", val_accuracies)
        plt.show()
    except Exception as e:
        traceback.print_exc()
        print("Error: Problems generating plot. See if a .png was generated in hw1/. "
              "If not, check the writeup and Piazza hw1p1 thread.")
if __name__ == "__main__":
    main()
