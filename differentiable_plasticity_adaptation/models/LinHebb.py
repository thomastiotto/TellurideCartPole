import torch
import math
import torch.nn as nn
import numpy as np

class LinHebb(nn.Module):
    """ Custom Linear layer with hebbian component """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        
        weights = torch.Tensor(size_out, size_in)
        alphas = torch.Tensor(size_out, size_in)
        bias = torch.Tensor(size_out)

        self.hebbs = torch.zeros((size_out, size_in), requires_grad = False)

        # nn.Parameter is a Tensor that's a module parameter.
        self.weights = nn.Parameter(weights)
        self.alphas = nn.Parameter(alphas)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        nn.init.kaiming_uniform_(self.alphas, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):

        w_times_x = torch.mm(x, self.weights.t() + torch.mul(self.alphas.t(), self.hebbs.t()))

        pre_y = x.view(x.shape[1], x.shape[0])
        post_y = w_times_x.expand(x.shape[1], w_times_x.shape[1])

        self.hebbs = (1.0 - 0.01) * self.hebbs + 0.01 * (pre_y * post_y)

        return torch.add(w_times_x, self.bias)  # w times x + b