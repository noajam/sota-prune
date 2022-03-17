import os
from tqdm import tqdm

import torch.nn as nn

from shrinkbench.experiment import PruningExperiment
import shrinkbench.models as models
from shrinkbench.plot import df_from_results, plot_df

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
clear_output()


os.environ['DATAPATH'] = 'data'

for strategy in ['LayerLAP', 'GlobalLAP', 'LayerMagNoBias', 'GlobalMagNoBias', 'OptimalBrainDamage']:
    for c in [2,4,6,8,10,12,14]:
        exp = PruningExperiment(dataset='CIFAR10', 
                                model='MnistNet',
                                strategy=strategy,
                                compression=c,
                                train_kwargs={'epochs':5},
                                pretrained=False)

        exp.run()
        clear_output()
        

df = df_from_results('results')
plot_df(df, 'compression', 'pre_acc5', markers='strategy', line='--', colors='strategy', suffix=' - pre')
plot_df(df, 'compression', 'post_acc5', markers='strategy', fig=False, colors='strategy')

plot_df(df, 'speedup', 'post_acc5', colors='strategy', markers='strategy')
# plt.yscale('log')
#plt.ylim(0.996,0.9995)
plt.xticks(2**np.arange(7))
plt.gca().set_xticklabels(map(str, 2**np.arange(7)))

df['compression_err'] = (df['real_compression'] - df['compression'])/df['compression']
plot_df(df, 'compression', 'compression_err', colors='strategy', markers='strategy')