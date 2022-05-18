from numpy import ndarray
from torch import Tensor

from shrinkbench.pruning import VisionPruning
from shrinkbench.strategies import map_importances, flatten_importances, fraction_threshold, importance_masks

import torch


# noinspection PyMethodOverriding
class LAMP(VisionPruning):
    def model_masks(self):
        importances = map_importances(self.lamp_importances, self.params())
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks


    def lamp_importances(self, param: ndarray) -> ndarray:
        return self._normalize_scores(param ** 2).numpy()


    def _normalize_scores(self, scores: ndarray) -> Tensor:
        """
        Normalizing scheme for LAMP.
        """
        #print(type(scores))
        
        scores = torch.tensor(scores)
        # sort scores in an ascending order
        sorted_scores, sorted_idx = scores.view(-1).sort(descending=False)
        
        # compute cumulative sum
        scores_cumsum_temp = sorted_scores.cumsum(dim=0)
        scores_cumsum = torch.zeros(scores_cumsum_temp.shape, device=scores.device)
        scores_cumsum[1:] = scores_cumsum_temp[:len(scores_cumsum_temp) - 1]
        
        # normalize by cumulative sum
        sorted_scores /= (scores.sum() - scores_cumsum)
        
        # tidy up and output
        new_scores = torch.zeros(scores_cumsum.shape, device=scores.device)
        new_scores[sorted_idx] = sorted_scores
        new_scores = new_scores.view(scores.shape)
        return new_scores