import os
from tqdm import tqdm

import torch.nn as nn

from shrinkbench.experiment import PruningExperiment
import shrinkbench.models as models
from shrinkbench.plot import df_from_results, plot_df

import matplotlib.pyplot as plt
import numpy as np

def main():
    os.environ['DATAPATH'] = 'data:large_data'
    #compressions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    #compressions = [10**x for x in np.linspace(0, 2, 11)] # compression ratios too high, cause error
    compressions = [1,2,4,8,16,32,64]
    #compressions = [1, 16] # for testing funtionality on model over all methods
    strategies = ['OptimalBrainDamage', 'LAMP', 'SNIP', 'GraSP', 'SynFlow', 'GlobalLAP', 'LayerLAP', 
                  'GlobalMagWeight', 'LayerMagWeight',
                  'GlobalMagGrad', 'LayerMagGrad', 'GlobalMagAct', 'LayerMagAct', 'RandomPruning']
    #strategies = ['SynFlow'] # for testing a particular method
    """
    Strategies to include in final result
    Random
    Magnitude
    Gradient Magnitude
    OBD (only with basic CNN network)
    Lookahead
    Layer-wise OBS
    SNIP
    GraSP
    SynFlow
    LAMP
    """
    for strategy in strategies:
        for c in compressions:
            exp = PruningExperiment(dataset='MNIST', 
                                    model='MnistNet',
                                    strategy=strategy,
                                    compression=c,
                                    train_kwargs={'epochs': 5},
                                    pretrained=False)
    
            exp.run()
            
    
    df = df_from_results('results')
    #plot_df(df, 'compression', 'pre_acc5', markers='strategy', line='--', colors='strategy', suffix=' - pre')
    plot_df(df, 'compression', 'post_acc5', markers='strategy', colors='strategy')
    
    plot_df(df, 'speedup', 'post_acc5', colors='strategy', markers='strategy')
    # plt.yscale('log')
    #plt.ylim(0.996,0.9995)
    plt.xticks(2**np.arange(7))
    plt.gca().set_xticklabels(map(str, 2**np.arange(7)))
    
    df['compression_err'] = (df['real_compression'] - df['compression'])/df['compression']
    plot_df(df, 'compression', 'compression_err', colors='strategy', markers='strategy')
    
    
    # TODO
    """
        - |DONE| Remove bias layers from pruning computations
        - !|NEXT|! Remove bias layers from metric computation, if it's not already absent
        - !|NEXT|! Find FLOP comptuation overflow problem
        - Fix models to be simpler (no batchnorms, use sequential forward)
        - |DONE| Implement SNIP (add Xavier normal init)
        - |Not a priority| Implement Layer-wise OBS
        - |DONE| Test LAMP
        - |DONE| Implement GraSP (add Xavier normal init)
        - |DONE| Implement SynFlow
        - |PENDING| Implement Layer-wise Versions of SNIP, GraSP, SynFlow
    """
    
if __name__ == '__main__':
    experiment_name = 'results_resnet_cifar10'
    acc_type = 'post_acc1'
    df = df_from_results(experiment_name)
    #plot_df(df, 'compression', 'pre_acc1', markers='strategy', line='--', colors='strategy', suffix=' - pre')
    plot_df(df, 'compression', acc_type, markers='strategy', colors='strategy')
    plt.savefig('graphs/' + experiment_name.replace('results_', '') + '_' + acc_type + '_compression.pdf', dpi=300, bbox_inches='tight', transparent=True)
    
    plot_df(df, 'speedup', acc_type, colors='strategy', markers='strategy')
    #plt.yscale('log')
    #plt.ylim(0.996,0.9995)
    plt.xticks(2**np.arange(7))
    plt.gca().set_xticklabels(map(str, 2**np.arange(7)))
    plt.savefig('graphs/' + experiment_name.replace('results_', '') + '_' + acc_type + '_speedup.pdf', dpi=300, bbox_inches='tight', transparent=True)
    
    df['compression_err'] = (df['real_compression'] - df['compression'])/df['compression']
    plot_df(df, 'compression', 'compression_err', colors='strategy', markers='strategy')
    plt.savefig('graphs/' + experiment_name.replace('results_', '') + '_comp_err.pdf', dpi=300, bbox_inches='tight', transparent=True)
    #main()