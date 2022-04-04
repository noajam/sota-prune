"""Optimal Brain Damage pruning

Implements pruning strategy that prunes as follows

For a given tensor, sort by saliency and
keep only the _fraction_ with highest saliencies
so that overall desired compression is achieved
"""

import numpy as np
import time
import random
from backpack import extend, backpack
from backpack.extensions import DiagHessian

from torch.utils.data import Sampler, DataLoader
import torch.nn as nn

from ..pruning import (LayerPruning,
                       VisionPruning,
                       GradientMixin,
                       ActivationMixin,
                       LinearMasked,
                       Conv2dMasked)
from .utils import (fraction_threshold,
                    fraction_mask,
                    map_importances,
                    flatten_importances,
                    importance_masks,
                    activation_importance)
from .. import models

import torchvision

class OptimalBrainDamage(VisionPruning):
    def model_masks(self):
        self.hessians = self.compute_hessians()
        importances = self.map_importances_obd(self.compute_saliency)
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks
        
        
        
    def compute_saliency(self, parameter, index):
        saliency = parameter * (self.hessians[index] ** 2) / 2
        return saliency
        
        
    def map_importances_obd(self, fn):
        return {module:
                {param: fn(importance, i)
                    for param, importance in params.items() if param == 'weight'}
                for i, (module, params) in enumerate(self.params().items())}
        

    def compute_hessians(self):
        self.model.train()
        
        batch_size, num_iterations = 60, 1000
        batch_sampler = BatchSampler(self.dataset, num_iterations, batch_size)
        data_loader = DataLoader(self.dataset, batch_sampler=batch_sampler, num_workers=4)
        
        model_seq = self.build_model('MnistNet')
        model_seq = self.copy_network(self.model, model_seq)
        
        criterion = nn.CrossEntropyLoss()
        criterion = extend(criterion)
        model_seq = extend(model_seq, use_converter=True).cuda()
        
        hessians = None
        for (x, y) in data_loader:
            x = x.cuda()
            y = y.cuda()
            
            out = model_seq(x)
            loss = criterion(out, y)
            
            with backpack(DiagHessian()):
                loss.backward()
                
            hessians = self.get_hessians(model_seq, hessians)
            
        return hessians
    
    
    def get_hessians(self, model, prev_hessians=None):
        if prev_hessians is None:
            flag = True
            prev_hessians = []
        else:
            flag = False
        
        cnt = 0
        
        prev_m = None
        for m in model.modules():
            if self.is_base_module(m) and not hasattr(m, 'is_classifier'):
                if not (isinstance(prev_m, nn.Conv2d) and isinstance(m, nn.Conv2d)):
                    if flag:
                        prev_hessians.append(m.weight.diag_h.data.cpu().detach().numpy())
                    else:
                        prev_hessians[cnt] += m.weight.diag_h.data.cpu().detach().numpy()
                        cnt += 1
        
                    m.weight.diag_h[:] = 0
        
            prev_m = m
        
        return prev_hessians
    

    def is_base_module(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            return True
        else:
            return False
        
        
    def is_masked_module(self, m):
    	if isinstance(m, LinearMasked) or isinstance(m, Conv2dMasked):
    		return True
    	else:
    		return False
    
    
    def is_batch_norm(self, m):
    	if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
    		return True
    	else:
    		return False
        
    
    def copy_network(self, network, network_seq):
        modules = self.extract_param_modules(network)
        modules_seq = self.extract_param_modules(network_seq)
    
        assert len(modules) == len(modules_seq)
    
        for i, (m, m_seq) in enumerate(zip(modules, modules_seq)):
            state_dict = m.state_dict()
            if self.is_masked_module(m):
                del state_dict['mask']
    
            if isinstance(m, nn.BatchNorm2d):
                assert isinstance(m_seq, nn.Conv2d)
                m_seq.bias.data = m.bias.data - m.running_mean.data / m.running_var.data.sqrt() * m.weight.data
                m_seq.weight.data = (m.weight.data / m.running_var.data.sqrt()).diag()
                m_seq.weight.data = m_seq.weight.data.view(m_seq.weight.shape[0], m_seq.weight.shape[1], 1, 1)
            else:
                m_seq.load_state_dict(state_dict)
    
        return network_seq
    
    
    def extract_param_modules(self, network):
        modules = []
    
        for m in network.modules():
            if self.is_base_module(m) or self.is_masked_module(m) or self.is_batch_norm(m):
                modules.append(m)
    
        return modules
    
    
    def build_model(self, model, pretrained=False, resume=None):
        if isinstance(model, str):
            if hasattr(models, model):
                model = getattr(models, model)(pretrained=pretrained)

            elif hasattr(torchvision.models, model):
                # https://pytorch.org/docs/stable/torchvision/models.html
                model = getattr(torchvision.models, model)(pretrained=pretrained)
                models.head.mark_classifier(model)  # add is_classifier attribute
            else:
                raise ValueError(f"Model {model} not available in custom models or torchvision models")

        return model

    
class OptimalBrainDamageLayer(OptimalBrainDamage, LayerPruning):
    def layer_masks(self, module):
        params = self.module_params(module)
        self.compute_hessians()
        ...
        # How to match hessians and layers?
        
        
    

    
class BatchSampler(Sampler):
	def __init__(self, dataset, num_iterations, batch_size):
		self.dataset = dataset
		self.num_iterations = num_iterations
		self.batch_size = batch_size

	def __iter__(self):
		for _ in range(self.num_iterations):
			indices = random.sample(range(len(self.dataset)), self.batch_size)
			yield indices

	def __len__(self):
		return self.num_iterations


