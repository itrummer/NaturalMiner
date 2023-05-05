'''
Created on Aug 21, 2021

Analyzes results for approaches summarizing item batches.

@author: immanueltrummer
'''
import argparse
import nminer.stats.common
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file', type=str, help='Path to input file')
    parser.add_argument('out_pre', type=str, help='Path prefix for output')
    args = parser.parse_args()
    
    df = pd.read_csv(args.in_file, sep=',')
    df['bquality'] = df['reward']
    df = nminer.stats.common.preprocess(df)
    df = df.loc[:,['nrfacts', 'nrpreds', 'approach', 'itemtime','fquality']]
    print(df.info())
    
    sum_agg = df.groupby(['nrfacts', 'nrpreds', 'approach'],
                         as_index=False).sum()
    avg_agg = df.groupby(['nrfacts', 'nrpreds', 'approach'],
                         as_index=False).mean()
    print(sum_agg)
    print(avg_agg)
    
    plt.rcParams.update({'text.usetex': True, 'font.size':9,
                         'font.serif':['Computer Modern'],
                         'font.family':'serif'})
    plt.close('all')
    
    nminer.stats.common.perf_breakdown(
        df, 'approach', ['SP', 'SC', 'CP'], 'Approach', 'itemtime', 
        None, 'Time (s)', 'log', f'{args.out_pre}timebreak.pdf')
    nminer.stats.common.perf_breakdown(
        df, 'approach', ['SP', 'SC', 'CP'], 'Approach', 'fquality',
        None, 'Quality', 'linear', f'{args.out_pre}qualbreak.pdf')