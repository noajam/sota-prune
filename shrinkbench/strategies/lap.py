"""Lookahead pruning

Implements pruning strategy that prunes as follows

For a given tensor, sort by absolute value and
keep only the _fraction_ with highest magnitudes
so that overall desired compression is achieved
"""

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

class GlobalLAP(VisionPruning):
    # Looks ahead and back at even non-prunable modules
    def model_masks(self):
        param_dict = self.params(only_prunable=True)
        depth = len(param_dict)
        
        self.layers_prev_list = [None] + list(range(depth - 1))
        self.layers_next_list = list(range(1, depth)) + [None]
        
        weights = []
        for param in param_dict:
            if 'weight' in param_dict[param]:
                weights.append(torch.from_numpy(param_dict[param]['weight']))
            
        importances = map_importances_lap(self.compute_score, self.params(), weights)
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        for module in param_dict:
            if 'bias' in param_dict[module]:
                masks[module]['bias'] = np.ones_like(param_dict[module]['bias'])
                
        return masks
    
    
    def compute_score(self, weights, layer):
        prev_score = _look_prev_score_multiple(weights, layer, self.layers_prev_list[layer])
        next_score = _look_next_score_multiple(weights, layer, self.layers_next_list[layer])
        
        score = (weights[layer] ** 2) * prev_score * next_score
        
        return np.array(score)


class LayerLAP(LayerPruning, VisionPruning):
    def layer_masks(self, module, layer):
        param_dict = self.params(only_prunable=True)
        depth = len(param_dict)
        
        self.layers_prev_list = [None] + list(range(depth - 1))
        self.layers_next_list = list(range(1, depth)) + [None]
        
        weights = []
        for param in param_dict:
            weights.append(torch.from_numpy(param_dict[param]['weight']))
        
        importances = {'weight': self.compute_score(weights, layer)}
        params = self.module_params(module)
        masks = {param: fraction_mask(importances[param], self.fraction)
                 for param, value in params.items() if value is not None and param=='weight'}
        if 'bias' in params:
            masks['bias'] = np.ones_like(params['bias'])
        return masks
    
    
    def compute_score(self, weights, layer):
        prev_score = _look_prev_score_multiple(weights, layer, self.layers_prev_list[layer])
        next_score = _look_next_score_multiple(weights, layer, self.layers_next_list[layer])
        
        score = (weights[layer] ** 2) * prev_score * next_score
        
        return np.array(score)
        
def map_importances_lap(fn, importances, weights):
    return {module:
            {param: fn(weights, i)
                for param, importance in params.items() if param=='weight'}
            for i, (module, params) in enumerate(importances.items())}
    
    
def _look_prev_score_multiple(weights, layer, layers_prev, bn_factors=None):
	if not isinstance(layers_prev, list):
		layers_prev = [layers_prev]

	if layers_prev[0] is None:  # layers_prev is None or [None]
		return 1

	prev_score = 0
	for layer_prev in layers_prev:
		if layer_prev == 'identity':
			prev_score += 1
			continue

		score = _look_prev_score(weights[layer], weights[layer_prev])
		if bn_factors is not None:
			score = _apply_bn_factor_prev(score, bn_factors[layer_prev])  # BN of prev layer
		prev_score += score

	return prev_score


def _look_next_score_multiple(weights, layer, layers_next, bn_factors=None):
	if not isinstance(layers_next, list):
		layers_next = [layers_next]

	if layers_next[0] is None:  # layers_next is None or [None]
		return 1

	next_score = 0
	for layer_next in layers_next:
		if layer_next == 'identity':
			next_score += 1
			continue

		score = _look_next_score(weights[layer], weights[layer_next])
		if bn_factors is not None:
			score = _apply_bn_factor_next(score, bn_factors[layer])  # BN of current layer
		next_score += score

	return next_score


def _look_prev_score(weight, weight_prev):
	wp_squared = weight_prev ** 2

	if weight.dim() == 2 and weight_prev.dim() == 2:
		wp_squared_sum = wp_squared.sum(dim=1)
		wp_squared_sum_mat = wp_squared_sum.view(1, -1).repeat(weight.size(0), 1)

	elif weight.dim() == 4 and weight_prev.dim() == 4:
		wp_squared_sum = wp_squared.sum(dim=3).sum(dim=2).sum(dim=1)
		wp_squared_sum_mat = wp_squared_sum.view(1, -1, 1, 1).repeat(weight.size(0), 1, weight.size(2), weight.size(3))

	elif weight.dim() == 2 and weight_prev.dim() == 4:
		if weight.size(1) == weight_prev.size(0):
			wp_squared_sum = wp_squared.sum(dim=3).sum(dim=2).sum(dim=1)
			wp_squared_sum_mat = wp_squared_sum.view(1, -1).repeat(weight.size(0), 1)

		elif (weight.size(1) % weight_prev.size(0)) == 0:
			output_per_channel = weight.size(1) // weight_prev.size(0)
			wp_squared_sum = wp_squared.sum(dim=3).sum(dim=2).sum(dim=1)
			wp_squared_sum = wp_squared_sum.view(-1, 1).repeat(1, output_per_channel)
			wp_squared_sum = wp_squared_sum.view(-1)
			wp_squared_sum_mat = wp_squared_sum.view(1, -1).repeat(weight.size(0), 1)

		else:
			raise NotImplementedError
	else:
		raise NotImplementedError

	return wp_squared_sum_mat


def _look_next_score(weight, weight_next):
	wn_squared = weight_next ** 2

	if weight.dim() == 2 and weight_next.dim() == 2:
		wn_squared_sum = wn_squared.sum(dim=0)
		wn_squared_sum_mat = wn_squared_sum.view(-1, 1).repeat(1, weight.size(1))

	elif weight.dim() == 4 and weight_next.dim() == 4:
		wn_squared_sum = wn_squared.sum(dim=3).sum(dim=2).sum(dim=0)
		wn_squared_sum_mat = wn_squared_sum.view(-1, 1, 1, 1).repeat(1, weight.size(1), weight.size(2), weight.size(3))

	elif weight.dim() == 4 and weight_next.dim() == 2:
		if weight.size(0) == weight_next.size(1):
			wn_squared_sum = wn_squared.sum(dim=0)
			wn_squared_sum_mat = wn_squared_sum.view(-1, 1, 1, 1).repeat(1, weight.size(1), weight.size(2), weight.size(3))

		elif (weight_next.size(1) % weight.size(0)) == 0:
			output_per_channel = weight_next.size(1) // weight.size(0)
			wn_squared_sum = wn_squared.sum(dim=0)
			wn_squared_sum = wn_squared_sum.view(-1, output_per_channel)
			wn_squared_sum = wn_squared_sum.sum(dim=1)
			wn_squared_sum_mat = wn_squared_sum.view(-1, 1, 1, 1).repeat(1, weight.size(1), weight.size(2), weight.size(3))

		else:
			raise NotImplementedError
	else:
		raise NotImplementedError

	return wn_squared_sum_mat


def _apply_bn_factor_prev(score, bn_factor_prev=None):
	if bn_factor_prev is not None:
		if score.shape[1] == bn_factor_prev.shape[0]:
			for idx in range(score.size(1)):
				score[:, idx] *= bn_factor_prev[idx] ** 2
		elif (score.shape[1] % bn_factor_prev.shape[0]) == 0:
			for idx in range(score.size(1)):
				score[:, idx] *= bn_factor_prev[idx // bn_factor_prev.shape[0]] ** 2
		else:
			raise NotImplementedError

	return score


def _apply_bn_factor_next(score, bn_factor_next=None):
	if bn_factor_next is not None:
		for idx in range(score.size(0)):
			score[idx] *= bn_factor_next[idx] ** 2

	return score