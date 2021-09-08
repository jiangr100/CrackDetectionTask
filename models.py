import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
from utils import *
import copy
from pathlib import Path

import torch.nn.functional as F


class Gradcam(nn.Module):

    def __init__(self, encoder_model, initializer, n_class=2):

        super(Gradcam, self).__init__()

        self.net = encoder_model()
        self.net.fc = nn.Linear(2048, n_class)
        init_weights = WeightInit(initializer=initializer, nonlinearity='relu', bias=False)
        self.net.apply(init_weights)

        self.n_class = n_class

        target_layer = self.net.layer4

        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        self.loss_func = nn.CrossEntropyLoss()


    def forward(self, input, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input['img'].size()

        logit = self.model_arch(input)
        score = self.loss_func(logit[:, 0].squeeze())

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        #alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit









