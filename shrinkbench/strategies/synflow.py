"""SynFlow pruning

Implements pruning strategy that prunes as follows

For a given tensor, sort by absolute value and
keep only the _fraction_ with highest magnitudes
so that overall desired compression is achieved
"""
from collections import OrderedDict, defaultdict

import numpy as np
import torch

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


class SynFlow(GradientMixin, VisionPruning):

    def model_masks(self):
        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])
        
        self.model.train()
        signs = linearize(self.model)

        input_dim = list(self.inputs[0,:].shape)
        input = torch.ones([1] + input_dim)#, dtype=torch.float64).to(device)
        output = self.model(input)
        torch.sum(output).backward()
        
        nonlinearize(self.model, signs)
        
        training = self.model.training
        
        gradients = defaultdict(OrderedDict)
        for module in self.model.modules():
            assert module not in gradients
            for name, param in module.named_parameters(recurse=False):
                if param.requires_grad and param.grad is not None:
                    gradients[module][name] = param.grad.detach().cpu().numpy().copy()
        
        self.model.zero_grad()
        self.model.train(training)
        
        params = self.params()

        importances = {mod:
                       {p: np.abs(gradients[mod][p] * params[mod][p])
                        for p in mod_params}
                       for mod, mod_params in params.items()}
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks

