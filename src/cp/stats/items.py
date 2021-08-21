'''
Created on Aug 13, 2021

Analyzes results of summarization for single items.

@author: immanueltrummer
'''
import argparse
import cp.stats.common
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Analyze CP benchmark.')
    parser.add_argument('results_path', type=str, help='tsv file with results')
    args = parser.parse_args()
    
    df = pd.read_csv(args.results_path, sep='\t')
    df = cp.stats.common.preprocess(df)
    print(df.info())

    plt.rcParams.update({'text.usetex': True, 'font.size':9})
    plt.close('all')
    
    alg_groups = {'gen': ['r', 'R', 'G', 'P'], 'rl': ['B', 'P', 'SB', 'SP']}
    
    for scenario in range(4):
        scenario_data = df.query(f'scenario == {scenario}')
        path_prefix = f'plots/S{scenario}_'
        
        for g_name, algs in alg_groups.items():
            file_path = path_prefix + f'{g_name}_qual.pdf'
            cp.stats.common.perf_breakdown(
                scenario_data, algs, 'fquality', 
                (-1.1, 1.1), 'Quality', 'linear', file_path)
            
            file_path = path_prefix + f'{g_name}_time.pdf'
            cp.stats.common.perf_breakdown(
                scenario_data, algs, 'time', 
                None, 'Time (s)', 'log', file_path)
    
    for g_name, algs in alg_groups.items():
        file_path = f'plots/sx_{g_name}_time.pdf'
        cp.stats.common.perf_plot(
            df, algs, 'time', None, 
            'Time (s)', 'log', file_path)
        file_path = f'plots/sx_{g_name}_quality.pdf'
        cp.stats.common.perf_plot(
            df, algs, 'fquality', (-1.1,1.1), 
            'Quality', 'linear', file_path)