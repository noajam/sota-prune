"""GraSP pruning

Implements pruning strategy that prunes as follows

For a given tensor, sort by absolute value and
keep only the _fraction_ with highest magnitudes
so that overall desired compression is achieved
"""
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn

from ..pruning import (LayerPruning,
                       VisionPruning,
                       GradientMixin,
                       ActivationMixin)
from .utils import (fraction_threshold,
                    fraction_mask,
                    map_importances,
                    flatten_importances,
                    importance_masks,
                    activation_importance)


class GraSP(GradientMixin, VisionPruning):

    def model_masks(self):
        params = self.params()
        grads = self.param_gradients()
        
        self.temp = 200
        self.eps = 1e-10
        self.model.train()
        
        loss_func = nn.CrossEntropyLoss()
        stopped_grads = 0
        
        pred = self.model(self.inputs) / self.temp
        loss = loss_func(pred, self.outputs)
        
        grads = torch.autograd.grad(loss, [p for (module, p) in self.model.named_parameters()], create_graph=False)
        flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
        stopped_grads += flatten_grads
        
        
        pred = self.model(self.inputs) / self.temp
        loss = loss_func(pred, self.outputs)
        
        grads = torch.autograd.grad(loss, [p for (module, p) in self.model.named_parameters()], create_graph=True)
        flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
        gnorm = (stopped_grads * flatten_grads).sum()
        gnorm.backward()

        training = self.model.training
        
        gradients = defaultdict(OrderedDict)
        for module in self.model.modules():
            assert module not in gradients
            for name, param in module.named_parameters(recurse=False):
                if param.requires_grad and param.grad is not None:
                    gradients[module][name] = param.grad.detach().cpu().numpy().copy()
        
        self.model.zero_grad()
        self.model.train(training)
        
        # Calculate score: Hg * theta (make theta negative to remove highest scores)
        
        
        importances = {mod:
                       {p: (gradients[mod][p] * -params[mod][p])
                        for p in mod_params}
                       for mod, mod_params in params.items()}
        flat_importances = flatten_importances(importances)
        norm_factor = np.abs(np.sum(flat_importances)) + self.eps
        importances = {mod:
                       {p: ((gradients[mod][p] * -params[mod][p]) / norm_factor)
                        for p in mod_params}
                       for mod, mod_params in params.items()}
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks

