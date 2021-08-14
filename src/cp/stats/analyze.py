'''
Created on Aug 13, 2021

@author: immanueltrummer
'''
import argparse
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.pyplot import ylabel

def perf_plot(df, approaches, metric):
    """ Plots a multi-plot for a subset of approaches.
    
    Args:
        df: data frame containing all benchmark results
        approaches: include those approaches in the plot
        metric: y axis summarizes this metric
    
    Returns:
        generated multi-plot
    """
    df = df.query(f'approach in {approaches}')
    df = df.loc[:,['approach', 'nrfacts', 'nrpreds', metric]]
    means = df.groupby(['approach', 'nrfacts', 'nrpreds']).mean()
    
    _, axes = plt.subplots(nrows=3, ncols=3, figsize=(3.5,3.5), 
                           subplotpars=matplotlib.figure.SubplotParams(wspace=0.5, hspace=0.75))
    for x_pos in range(3):
        for y_pos in range(3):
            nr_facts = x_pos + 1
            nr_preds = y_pos + 1
            q = f'nrfacts == {nr_facts} and nrpreds == {nr_preds}'
            ax = axes[y_pos, x_pos]
            
            ax.set_ylim(ymin=-1, ymax=1)
            ax.set_title(f'{nr_facts}F, {nr_preds}P', fontsize=9)
            if x_pos == 0:
                ax.set_ylabel('Quality')

            cur_means = means.query(q)
            approaches = cur_means.index.get_level_values(0)
            ax.bar(approaches, cur_means[metric])
    
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Analyze CP benchmark.')
    parser.add_argument('results_path', type=str, help='tsv file with results')
    args = parser.parse_args()
    
    df = pd.read_csv(args.results_path, sep='\t')
    methods_short = {
        'rand1':'R1', 'rand':'RM', 'gen':'TG', 'rlempty':'CP',
        'sample':'AP', 'rlcube':'CB', 'rlproactive':'PC'}
    df['approach'] = df['approach'].apply(lambda a:methods_short[a])
    df['fquality'] = df['bquality'].apply(lambda q:max(q,-1))
    print(df.info())
    
    plt.rcParams.update({'text.usetex': True})
    plt.close('all')
    perf_plot(df, ['R1', 'RM', 'TG', 'CP'], 'fquality')