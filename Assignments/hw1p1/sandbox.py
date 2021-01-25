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
from mytorch.optim.adam import Adam


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
    test_dropout_forward()
    test_linear_adam()
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

##########-----Bonus 
# Adam tests
def test_linear_adam():
    check_torch_version()
    np.random.seed(11785)
    mytorch_mlp = Sequential(Linear(10, 20))
    mytorch_optimizer = Adam(mytorch_mlp.parameters())
    mytorch_criterion = CrossEntropyLoss()
    return test_step(mytorch_mlp, mytorch_optimizer, 5, 5,
                            mytorch_criterion=mytorch_criterion)


def test_step(mytorch_model, mytorch_optimizer, train_steps, eval_steps,
              mytorch_criterion=None, batch_size=(2, 5)):
    """
    Tests subsequent forward, back, and update operations, printing whether
    a mismatch occurs in forward or backwards.

    Returns whether the test succeeded.
    """
    pytorch_model = get_same_pytorch_mlp(mytorch_model)
    pytorch_optimizer = get_same_pytorch_optimizer(
        mytorch_optimizer, pytorch_model)
    pytorch_criterion = get_same_pytorch_criterion(mytorch_criterion)
    batch_size = np.random.randint(*batch_size) if type(batch_size) == tuple\
        else batch_size
    x, y = generate_dataset_for_mytorch_model(mytorch_model, batch_size)

    mytorch_model.train()
    pytorch_model.train()
    for s in range(train_steps):
        pytorch_optimizer.zero_grad()
        mytorch_optimizer.zero_grad()

        forward_passed, (mx, my, px, py) = \
            forward_(mytorch_model, mytorch_criterion,
                     pytorch_model, pytorch_criterion, x, y)
        if not forward_passed:
            print("Forward failed")
            return False

        backward_passed = backward_(
            mx, my, mytorch_model, px, py, pytorch_model)
        if not backward_passed:
            print("Backward failed")
            return False

        pytorch_optimizer.step()
        mytorch_optimizer.step()
        # check that model is correctly configured
        check_model_param_settings(mytorch_model)

    mytorch_model.eval()
    pytorch_model.eval()
    for s in range(eval_steps):
        pytorch_optimizer.zero_grad()
        mytorch_optimizer.zero_grad()

        forward_passed, (mx, my, px, py) = \
            forward_(mytorch_model, mytorch_criterion,
                     pytorch_model, pytorch_criterion, x, y)
        if not forward_passed:
            print("Forward failed")
            return False

    # Check that each weight tensor is still configured correctly
    try:
        for param in mytorch_model.parameters():
            assert param.requires_grad, "Weights should have requires_grad==True!"
            assert param.is_leaf, "Weights should have is_leaf==True!"
            assert param.is_parameter, "Weights should have is_parameter==True!"
    except Exception as e:
        traceback.print_exc()
        return False

    return True
def get_same_pytorch_optimizer(mytorch_optimizer, pytorch_mlp):
    """
    Returns a pytorch optimizer matching the given mytorch optimizer, except
    with the pytorch mlp parameters, instead of the parametesr of the mytorch
    mlp
    """
    lr = mytorch_optimizer.lr
    betas = mytorch_optimizer.betas
    eps = mytorch_optimizer.eps
    return torch.optim.Adam(pytorch_mlp.parameters(), lr=lr, betas=betas, eps=eps)
def backward_(mytorch_x, mytorch_y, mytorch_model, pytorch_x, pytorch_y, pytorch_model):
    """
    Calls backward on both mytorch and pytorch outputs, and returns whether
    computed gradients match.
    """
    mytorch_y.backward()
    pytorch_y.sum().backward()
    # check that model is correctly configured
    check_model_param_settings(mytorch_model)
    return check_gradients(mytorch_x, pytorch_x, mytorch_model, pytorch_model)

def check_gradients(mytorch_x, pytorch_x, mytorch_model, pytorch_model):
    """
    Checks computed gradients, assuming forward has already occured.

    Checked gradients are the gradients of linear weights and biases, and the
    gradient of the input.
    """

    if not assertions_all(mytorch_x.grad.data, pytorch_x.grad.detach().numpy(), 'dx'):
        return False
    mytorch_linear_layers = get_mytorch_linear_layers(mytorch_model)
    pytorch_linear_layers = get_pytorch_linear_layers(pytorch_model)
    for mytorch_linear, pytorch_linear in zip(mytorch_linear_layers, pytorch_linear_layers):
        pytorch_dW = pytorch_linear.weight.grad.detach().numpy()
        pytorch_db = pytorch_linear.bias.grad.detach().numpy()
        mytorch_dW = mytorch_linear.weight.grad.data
        mytorch_db = mytorch_linear.bias.grad.data

        if not assertions_all(mytorch_dW, pytorch_dW, 'dW'):
            return False
        if not assertions_all(mytorch_db, pytorch_db, 'db'):
            return False
    return True
    
def check_torch_version():
    """Checks that Torch is of the correct version for this assignment.
    The official torch's implementation of `Adam` was bugged until they fixed it in 1.3.0.

    You will be implementing the correct version; thus you need at least torch version 1.3.0.
    """
    min_required_torch_version = "1.3.0"
    local_torch_version = torch.__version__
    # local_torch_version= "1.2.9" # for debugging purposes

    # Compare version strings
    if int(local_torch_version[0]) < int(min_required_torch_version[0]):
        valid_version = False
    elif int(local_torch_version[0]) == int(min_required_torch_version[0]) and \
        int(local_torch_version[2]) < int(min_required_torch_version[2]):
        valid_version = False
    else:
        valid_version = True

    if not valid_version:
        print("*****************************************************************************************")
        print("***ERROR: You must upgrade to torch version >= 1.3.0 (ideally update to the latest version).\n\t"
                        "Until version 1.3.0, the official torch had a bugged implementation of Adam and AdamW.\n\t"
                        "You will be implementing the correct version, and thus will need torch >= 1.3.0.\n\t"
                        "Autolab will have version >= 1.3.0 of torch as well.\n\t"
                        "If you do not upgrade, the local autograder will NOT work properly.\n\t"
                        "Assume that future homeworks won't be affected by torch version issues.")

def forward_(mytorch_model, mytorch_criterion, pytorch_model,
             pytorch_criterion, x, y):
    """
    Calls forward on both mytorch and pytorch models.

    x: ndrray (batch_size, in_features)
    y: ndrray (batch_size,)

    Returns (passed, (mytorch x, mytorch y, pytorch x, pytorch y)),
    where passed is whether the test passed

    """
    # forward
    pytorch_x = Variable(torch.tensor(x).double(), requires_grad=True)
    pytorch_y = pytorch_model(pytorch_x)
    if not pytorch_criterion is None:
        pytorch_y = pytorch_criterion(pytorch_y, torch.LongTensor(y))
    mytorch_x = Tensor(x, requires_grad=True)
    mytorch_y = mytorch_model(mytorch_x)
    if not mytorch_criterion is None:
        mytorch_y = mytorch_criterion(mytorch_y, Tensor(y))

    # check that model is correctly configured
    check_model_param_settings(mytorch_model)

    # forward check
    if not assertions_all(mytorch_y.data, pytorch_y.detach().numpy(), 'y'):
        return False, (mytorch_x, mytorch_y, pytorch_x, pytorch_y)

    return True, (mytorch_x, mytorch_y, pytorch_x, pytorch_y)




# Dropout tests
def test_dropout_forward():
    np.random.seed(11785)
    
    # run on small model forward only
    x = Tensor.randn(5, 10)
    model = Sequential(Linear(10, 5), ReLU(), Dropout(p=0.6))
    my_output = model(x)
    
    # check that model is correctly configured
    check_model_param_settings(model)

    test_output = load_numpy_array('autograder/hw1_bonus_autograder/outputs/dropout_forward.npy')
    return assertions_all(my_output.data, test_output, "test_dropout_forward", 1e-5, 1e-6)

def test_dropout_forward_backward():
    np.random.seed(11785)
    
    # run on small model, forward backward (no step)
    model = Sequential(Linear(10, 20), ReLU(), Dropout(p=0.6))
    x, y = generate_dataset_for_mytorch_model(model, 5)
    x, y = Tensor(x), Tensor(y)
    criterion = CrossEntropyLoss()
    out = model(x)
    
    # check that model is correctly configured
    check_model_param_settings(model)
    
    test_out = load_numpy_array('autograder/hw1_bonus_autograder/outputs/backward_output.npy')
    
    if not assertions_all(out.data, test_out, "test_dropout_forward_backward_output", 1e-5, 1e-6):
        return False
    
    loss = criterion(out, y)
    loss.backward()
    
    # check that model is correctly configured
    check_model_param_settings(model)
    
    assert model[0].weight.grad is not None, "Linear layer must have gradient."
    assert model[0].weight.grad.grad is None, "Final gradient tensor must not have its own gradient"
    assert model[0].weight.grad.grad_fn is None, "Final gradient tensor must not have its own grad function"
    assert model[0].weight.requires_grad, "Weight tensor must have requires_grad==True"
    assert model[0].weight.is_parameter, "Weight tensor must be marked as a parameter tensor"

    test_grad = load_numpy_array('autograder/hw1_bonus_autograder/outputs/backward_grad.npy')
    
    return assertions_all(model[0].weight.grad.data, test_grad, "test_dropout_forward_backward_grad", 1e-5, 1e-6)

def test_big_model_step():
    np.random.seed(11785)
    
    # run a big model
    model = Sequential(Linear(10, 15), ReLU(), Dropout(p=0.2), 
                       Linear(15, 20), ReLU(), Dropout(p=0.1))
    x, y = generate_dataset_for_mytorch_model(model, 4)
    x, y = Tensor(x), Tensor(y)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
    
    # check output correct
    out = model(x)
    test_out = load_numpy_array('autograder/hw1_bonus_autograder/outputs/big_output.npy')

    # check that model is correctly configured
    check_model_param_settings(model)

    if not assertions_all(out.data, test_out, "test_big_model_step_out", 1e-5, 1e-6):
        return False
    
    # run backward
    loss = criterion(out, y)
    loss.backward()
    
    # check that model is correctly configured
    check_model_param_settings(model)
    
    # check params are correct (sorry this is ugly)
    assert model[0].weight.grad is not None, "Linear layer must have gradient."
    assert model[0].weight.grad.grad is None, "Final gradient tensor must not have its own gradient"
    assert model[0].weight.grad.grad_fn is None, "Final gradient tensor must not have its own grad function"
    assert model[0].weight.requires_grad, "Weight tensor must have requires_grad==True"
    assert model[0].weight.is_parameter, "Weight tensor must be marked as a parameter tensor"
    assert model[3].weight.grad is not None, "Linear layer must have gradient."
    assert model[3].weight.grad.grad is None, "Final gradient tensor must not have its own gradient"
    assert model[3].weight.grad.grad_fn is None, "Final gradient tensor must not have its own grad function"
    assert model[3].weight.requires_grad, "Weight tensor must have requires_grad==True"
    assert model[3].weight.is_parameter, "Weight tensor must be marked as a parameter tensor"
    
    # check gradient for linear layer at idx 0 is correct
    test_grad = load_numpy_array('autograder/hw1_bonus_autograder/outputs/big_grad.npy')
    if not assertions_all(model[0].weight.grad.data, test_grad, "test_big_model_grad_0", 1e-5, 1e-6):
        return False
    
    # check gradient for linear layer at idx 3 is correct
    test_grad = load_numpy_array('autograder/hw1_bonus_autograder/outputs/big_grad_3.npy')
    if not assertions_all(model[3].weight.grad.data, test_grad, "test_big_model_grad_3", 1e-5, 1e-6):
        return False

    # weight update with adam
    optimizer.step()
    
    # check updated weight values
    assert model[0].weight.requires_grad, "Weight tensor must have requires_grad==True"
    assert model[0].weight.is_parameter, "Weight tensor must be marked as a parameter tensor"

    test_weights_3 = load_numpy_array('autograder/hw1_bonus_autograder/outputs/big_weight_update_3.npy')
    test_weights_0 = load_numpy_array('autograder/hw1_bonus_autograder/outputs/big_weight_update_0.npy')
    
    return assertions_all(model[0].weight.data, test_weights_0, "test_big_weight_update_0", 1e-5, 1e-6) and \
        assertions_all(model[3].weight.data, test_weights_3, "test_big_weight_update_3", 1e-5, 1e-6)

##############################
# Utilities for testing MLPs #
##############################

def generate_dataset_for_mytorch_model(mytorch_model, batch_size):
    """
    Generates a fake dataset to test on.

    Returns x: ndarray (batch_size, in_features),
            y: ndarray (batch_size,)
    where in_features is the input dim of the mytorch_model, and out_features
    is the output dim.
    """
    in_features = get_mytorch_model_input_features(mytorch_model)
    out_features = get_mytorch_model_output_features(mytorch_model)
    x = np.random.randn(batch_size, in_features)
    y = np.random.randint(out_features, size=(batch_size,))
    return x, y

def get_mytorch_model_input_features(mytorch_model):
    """
    Returns in_features for the first linear layer of a mytorch
    Sequential model.
    """
    return get_mytorch_linear_layers(mytorch_model)[0].in_features


def get_mytorch_model_output_features(mytorch_model):
    """
    Returns out_features for the last linear layer of a mytorch
    Sequential model.
    """
    return get_mytorch_linear_layers(mytorch_model)[-1].out_features


def get_mytorch_linear_layers(mytorch_model):
    """
    Returns a list of linear layers for a mytorch model.
    """
    return list(filter(lambda x: isinstance(x, Linear), mytorch_model.layers))


def get_pytorch_linear_layers(pytorch_model):
    """
    Returns a list of linear layers for a pytorch model.
    """
    return list(filter(lambda x: isinstance(x, nn.Linear), pytorch_model))

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
