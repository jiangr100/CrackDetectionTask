import torch
import torch.nn
import random
import time


class WeightInit:

    def __init__(self, initializer, nonlinearity, bias=False):
        self.initializer = initializer
        self.nonlinearity = nonlinearity
        self.bias = bias

    def __call__(self, m):
        if type(m) == torch.nn.Conv2d:
            self.initializer(m.weight, nonlinearity=self.nonlinearity)
            if self.bias:
                m.bias.data.fill_(0)