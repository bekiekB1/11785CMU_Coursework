import numpy as np

import mytorch.tensor as tensor
from mytorch.autograd_engine import Function


def unbroadcast(grad, shape, to_keep=0):
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor: {}".format(a.shape))
        requires_grad = a.requires_grad
        b = tensor.Tensor(a.data.T, requires_grad=requires_grad,
                                    is_leaf=not requires_grad)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.T)

class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))
        ctx.shape = a.shape
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.reshape(shape), requires_grad=requires_grad,
                                                 is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.reshape(ctx.shape)), None

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.log(a.data), requires_grad=requires_grad,
                                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(grad_output.data / a.data)

class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for exp must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.exp(a.data), requires_grad=requires_grad,
                                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(np.exp(a.data) * grad_output.data)

class Sqrt(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for exp must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.sqrt(a.data), requires_grad=requires_grad,
                                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(1/(2*np.sqrt(a.data)) * grad_output.data)

"""EXAMPLE: This represents an Op:Add node to the comp graph.

See `Tensor.__add__()` and `autograd_engine.Function.apply()`
to understand how this class is used.

Inherits from:
    Function (autograd_engine.Function)
"""
class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data + b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = np.ones(a.shape) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = np.ones(b.shape) * grad_output.data

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor') or \
            a.data.shape != b.data.shape:
            pass
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data - b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = np.ones(a.shape) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = np.ones(b.shape) * -grad_output.data

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor') or \
            a.data.shape != b.data.shape:
            pass
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data * b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout        
        grad_a = np.array(b.data) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = np.array(a.data) * grad_output.data
        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b

#@ could be overrided like + and -
class matmul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor') or \
            a.data.shape != b.data.shape:
            pass
        
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data @ b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout        
        grad_a = grad_output.data @ np.array(b.data).T
       
        # dL/db = dout/db * dL/dout
        grad_b = np.array(a.data).T @ grad_output.data
        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor') or \
            a.data.shape != b.data.shape:
            pass
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data / b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout        
        grad_a = np.array(1/b.data) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = np.array(-a.data/b.data**2) * grad_output.data
        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b


class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Only log of tensor is supported")
        ctx.axis = axis
        ctx.shape = a.shape
        if axis is not None:
            ctx.len = a.shape[axis]
        ctx.keepdims = keepdims
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.sum(axis = axis, keepdims = keepdims), \
                          requires_grad=requires_grad, is_leaf=not requires_grad)
        #print(a.shape, c.shape)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        grad_out = grad_output.data

        if (ctx.axis is not None) and (not ctx.keepdims):
            grad_out = np.expand_dims(grad_output.data, axis=ctx.axis)
        else:
            grad_out = grad_output.data.copy()

        grad = np.ones(ctx.shape) * grad_out

        assert grad.shape == ctx.shape
        # Take note that gradient tensors SHOULD NEVER have requires_grad = True.
        return tensor.Tensor(grad), None, None


class ReLU(Function):
    @staticmethod
    def forward(ctx, a):        
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.maximum(a.data,0), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a = ctx.saved_tensors[0]

        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout      
        grad_relu = grad_output.data * np.where(a.data<0,0,1)

        # the order of gradients returned should match the order of the arguments
        grad_relu = tensor.Tensor(unbroadcast(grad_relu, a.shape))
        return grad_relu


# TODO: Implement more Functions below


def cross_entropy(predicted, target):
    """Calculates Cross Entropy Loss (XELoss) between logits and true labels.
    For MNIST, don't call this function directly; use nn.loss.CrossEntropyLoss instead.

    Args:
        predicted (Tensor): (batch_size, num_classes) logits
        target (Tensor): (batch_size,) true labels

    Returns:
        Tensor: the loss as a float, in a tensor of shape ()
    """
    batch_size, num_classes = predicted.shape

    # Tip: You can implement XELoss all here, without creating a new subclass of Function.
    #      However, if you'd prefer to implement a Function subclass you're free to.
    #      Just be sure that nn.loss.CrossEntropyLoss calls it properly.

    # Tip 2: Remember to divide the loss by batch_size; this is equivalent
    #        to reduction='mean' in PyTorch's nn.CrossEntropyLoss
    a = np.amax(predicted.data, axis=1,keepdims=True)
    a = tensor.Tensor(a,requires_grad=False)
    logsoft = predicted - a - tensor.Tensor.log(tensor.Tensor.sum(tensor.Tensor.exp(predicted - a),axis=1,keepdims=True))
    target = to_one_hot(target,num_classes)
    nllLoss = tensor.Tensor(-1) * tensor.Tensor.sum(logsoft * target) / tensor.Tensor(batch_size)
    return nllLoss

def to_one_hot(arr, num_classes):
    """(Freebie) Converts a tensor of classes to one-hot, useful in XELoss

    Example:
    >>> to_one_hot(Tensor(np.array([1, 2, 0, 0])), 3)
    [[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0],
     [1, 0, 0]]

    Args:
        arr (Tensor): Condensed tensor of label indices
        num_classes (int): Number of possible classes in dataset
                           For instance, MNIST would have `num_classes==10`
    Returns:
        Tensor: one-hot tensor
    """
    arr = arr.data.astype(int)
    a = np.zeros((arr.shape[0], num_classes))
    a[np.arange(len(a)), arr] = 1
    return tensor.Tensor(a, requires_grad = True)

########################################-------HW1p2-----###################

def get_conv1d_output_size(input_size, kernel_size, stride):
    """Gets the size of a Conv1d output.

    Notes:
        - This formula should NOT add to the comp graph.
        - Yes, Conv2d would use a different formula,
        - But no, you don't need to account for Conv2d here.
        
        - If you want, you can modify and use this function in HW2P2.
            - You could add in Conv1d/Conv2d handling, account for padding, dilation, etc.
            - In that case refer to the torch docs for the full formulas.

    Args:
        input_size (int): Size of the input to the layer
        kernel_size (int): Size of the kernel
        stride (int): Stride of the convolution

    Returns:
        int: size of the output as an int (not a Tensor or np.array)
    """
    # TODO: implement the formula in the writeup. One-liner; don't overthink
    return ((input_size - kernel_size) // stride) + 1 # See python doc for full formula
    
    #raise NotImplementedError("TODO: Complete functional.get_conv1d_output_size()!")


class Conv1d(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride):
        """The forward/backward of a Conv1d Layer in the comp graph.
        
        Notes:
            - Make sure to implement the vectorized version of the pseudocode
            - See Lec 10 slides # TODO: FINISH LOCATION OF PSEUDOCODE
            - No, you won't need to implement Conv2d for this homework.
        
        Args:
            x (Tensor): (batch_size, in_channel, input_size) input data
            weight (Tensor): (out_channel, in_channel, kernel_size)
            bias (Tensor): (out_channel,)
            stride (int): Stride of the convolution
        
        Returns:
            Tensor: (batch_size, out_channel, output_size) output data
        """
        # For your convenience: ints for each size
        batch_size, in_channel, input_size = x.shape
        out_channel, _, kernel_size = weight.shape
        #Conv1d.stride = stride
        # TODO: Save relevant variables for backward pass
        ctx.save_for_backward(x,weight,tensor.Tensor(stride,requires_grad=False))

        # TODO: Get output size by finishing & calling get_conv1d_output_size()
        output_size = get_conv1d_output_size(input_size, kernel_size,stride)

        # TODO: Initialize output with correct size
        out = np.zeros(shape = (batch_size, out_channel, output_size))
        
        # TODO: Calculate the Conv1d output.
        # Remember that we're working with np.arrays; no new operations needed.
        for idx,t in enumerate(range(0,input_size-kernel_size+1,stride)):
            segment = x.data[:,:,t:t+kernel_size] 
            out[:,:,idx] = np.einsum('ijk,ljk->il',segment,weight.data) + bias.data
        return tensor.Tensor(out,requires_grad=True,is_leaf=False)
        # TODO: Put output into tensor with correct settings and return 
        #raise NotImplementedError("Implement functional.Conv1d.forward()!")


    
    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Finish Conv1d backward pass. It's surprisingly similar to the forward pass.
        
        x,weight,stride = ctx.saved_tensors
        batch_size, in_channel, input_size = x.shape
        out_channel, _, kernel_size = weight.shape
        stride = stride.data.item()
        output_size = grad_output.shape[-1]   

        #Or grad_w = np.zeros_like(weight) #since gradient is not transposed in pytorch contrary to math
        grad_w = np.zeros(shape=(out_channel,in_channel,kernel_size))
        grad_y = np.zeros(shape=(batch_size,in_channel,input_size))
        grad_b = np.zeros(shape=(out_channel,))

        #Grad_b
        grad_b = np.einsum('ijk->j',grad_output.data)
        
        ##grad_w
        #upsample
        new_shape = (grad_output.shape[2]-1) * stride + 1 
        
        out_z = np.zeros((grad_output.shape[0],grad_output.shape[1],new_shape))
        out_z[:,:,::stride] = grad_output.data

        for t in range(input_size-out_z.shape[2]+1):
            if t < grad_w.shape[2]:
                seg_b = x.data[:,:,t:t+out_z.shape[2]]
                grad_w[:,:,t] = np.einsum('ijk,ilk->lj',seg_b,out_z)

        ### grad_x
        #For strided z from y|upsample
        new_k = (grad_output.shape[2]-1) * stride + 1 
        out = np.zeros((grad_output.shape[0],grad_output.shape[1],new_k))
        out[:,:,::stride] = grad_output.data

        #Padding to restore kernel_size - 1 # but not for stride greater than 1, but we upsample and behave stride as 1
        npad = ((0, 0), (0, 0), (kernel_size-1, kernel_size-1))
        out_2 = np.pad(out,pad_width=npad, mode='constant', constant_values=0)
        flip_weight = np.flip(weight.data,axis=2)
        for idx,t in enumerate(range(out_2.shape[2]-kernel_size+1)):
            seg_a = out_2[:,:,t:t+kernel_size]
            grad_y[:,:,t] = np.einsum('ijk,jlk->il',seg_a,flip_weight)

        
        return tensor.Tensor(grad_y,requires_grad=True,is_leaf=True), tensor.Tensor(grad_w,requires_grad=True,is_leaf=True) \
        ,tensor.Tensor(grad_b,requires_grad=True,is_leaf=True)




class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        b_data = np.divide(1.0, np.add(1.0, np.exp(-a.data)))
        ctx.out = b_data[:]
        b = tensor.Tensor(b_data, requires_grad=a.requires_grad)
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        b = ctx.out
        grad = grad_output.data * b * (1-b)
        return tensor.Tensor(grad)
    
class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        b = tensor.Tensor(np.tanh(a.data), requires_grad=a.requires_grad)
        ctx.out = b.data[:]
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.out
        grad = grad_output.data * (1-out**2)
        return tensor.Tensor(grad)