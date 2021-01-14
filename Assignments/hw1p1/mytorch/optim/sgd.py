import sys
import numpy as np

from mytorch.optim.optimizer import Optimizer


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.
    
    >>> optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    Args:
        params (list or iterator): <some module>.parameters()
        lr (float): learning rate (eta)
        momentum (float): momentum factor (beta)

    Inherits from:
        Optimizer (optim.optimizer.Optimizer)
    """
    def __init__(self, params, lr=0.001, momentum=0.0):
        super().__init__(params) # inits parent with network params
        self.lr = lr
        self.momentum = momentum

        # This tracks the momentum of each weight in each param tensor
        self.momentums = [np.zeros(t.shape) for t in self.params]

    def step(self):
        """Updates params based on gradients stored in param tensors"""
        # SGD without momentum
        #for prm in self.params:
        #    prm.data -= self.lr * prm.grad.data

        #SGD with momentum  || For just SGD, self.momentum = 0.0
        for idx,prm in enumerate(self.params):
            self.momentums[idx] = self.momentum * self.momentums[idx] - self.lr * prm.grad.data
            prm.data += self.momentums[idx]
        #type(self.params[0])
        #self.momentums = [self.momentum*self.momentums[idx] - self.lr * prm.grad.data for idx,prm in enumerate(self.params)]
        #type(self.params[0])
        #self.params[:] = map(lambda x, y: x.data+y, self.params,self.momentums)
        #raise Exception("TODO: Implement SGD.step()")
