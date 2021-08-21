'''
Created on Aug 21, 2021

Analyzes results for approaches summarizing item batches.

@author: immanueltrummer
'''
import argparse
import cp.stats.common
import pandas as pd

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file', type=str, help='Path to input file')
    parser.add_argument('out_pre', type=str, help='Path prefix for output')
    args = parser.parse_args()
    
    df = pd.read_csv(args.in_file, sep=',')
    df['bquality'] = df['reward']
    df = cp.stats.common.preprocess(df)
    df = df.loc[:,['nrfacts', 'nrpreds', 'approach', 'itemtime','fquality']]
    print(df.info())
    
    sum_agg = df.groupby(['nrfacts', 'nrpreds', 'approach'], 
                         as_index=False).sum()
    avg_agg = df.groupby(['nrfacts', 'nrpreds', 'approach'], 
                         as_index=False).mean()
    print(sum_agg)
    print(avg_agg)
    
    cp.stats.common.perf_breakdown(
        sum_agg, ['SP', 'B', 'CB'], 'itemtime', 
        None, 'Time (s)', 'linear', f'{args.out_pre}timebreak.pdf')
    cp.stats.common.perf_breakdown(
        avg_agg, ['SP', 'B', 'CB'], 'fquality', 
        None, 'Quality', 'linear', f'{args.out_pre}qualbreak.pdf')