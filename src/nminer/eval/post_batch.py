'''
Created on Sep 4, 2021

@author: immanueltrummer
'''
import argparse
import pandas as pd

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str, help='path to input file')
    parser.add_argument('out_path', type=str, help='path to output file')
    args = parser.parse_args()
    
    df = pd.read_csv(args.in_path, sep=',')
    df.drop('text', axis=1, inplace=True)
    df.drop('cluster', axis=1, inplace=True)
    
    batch_df = df.query('approach in ["simple", "cluster"]')
    groups = batch_df.groupby(['nrfacts', 'nrpreds', 'pred'])
    aggs = groups.agg('max').reset_index()
    aggs['approach'] = 'bestof'
    
    all_df = pd.concat([df, aggs], ignore_index=True)
    all_df.to_csv(args.out_path, sep=',')