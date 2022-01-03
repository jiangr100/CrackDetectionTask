import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
from utils import *
import copy
from pathlib import Path
from focalloss import FocalLoss
from torch.nn import BCELoss

import torch.nn.functional as F


class Gradcam(nn.Module):

    def __init__(self, encoder_model, initializer, n_class=2):

        super(Gradcam, self).__init__()

        self.net = encoder_model()
        self.net.fc = nn.Linear(2048, n_class)
        init_weights = WeightInit(initializer=initializer, nonlinearity='relu', bias=False)
        self.net.apply(init_weights)

        self.softmax = torch.nn.Softmax()
        self.n_class = n_class
        self.Sigmoid = nn.Sigmoid()

        target_layer = self.net.layer4

        self.loss_func = BCELoss()

    def forward(self, input):
        # return self.softmax(self.net(input))
        return self.net(input)











