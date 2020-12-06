from mytorch.tensor import Tensor
import numpy as np
from mytorch.nn.module import Module

class BatchNorm1d(Module):
    """Batch Normalization Layer

    Args:
        num_features (int): # dims in input and output
        eps (float): value added to denominator for numerical stability
                     (not important for now)
        momentum (float): value used for running mean and var computation

    Inherits from:
        Module (mytorch.nn.module.Module)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features

        self.eps = Tensor(np.array([eps]))
        self.momentum = Tensor(np.array([momentum]))

        # To make the final output affine
        self.gamma = Tensor(np.ones((self.num_features,)), requires_grad=True, is_parameter=True)
        self.beta = Tensor(np.zeros((self.num_features,)), requires_grad=True, is_parameter=True)

        # Running mean and var
        self.running_mean = Tensor(np.zeros(self.num_features,), requires_grad=False, is_parameter=False)
        self.running_var = Tensor(np.ones(self.num_features,), requires_grad=False, is_parameter=False)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, num_features)
        Returns:
            Tensor: (batch_size, num_features)
        """
        if(self.is_train == False):
            norm1 = x - self.running_mean
            norm = norm1 / Tensor.sqrt(self.running_var + self.eps)
        else:
            sample_mean = Tensor.sum(x,axis=0) / Tensor(x.shape[0])
            x_sub_mean = x - sample_mean
            sample_var = Tensor.sum(x_sub_mean*x_sub_mean,axis=0) / Tensor(x.shape[0])
            norm1 = x-sample_mean
            norm = norm1 / Tensor.sqrt(sample_var + self.eps)
            self.running_mean = self.momentum * self.running_mean + (Tensor(1) - self.momentum) * sample_mean
            self.running_var = self.momentum * self.running_var + (Tensor(1) - self.momentum) * sample_var
        out = self.gamma * norm + self.beta
        return out
        #raise Exception("TODO!")
