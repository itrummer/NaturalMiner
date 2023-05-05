'''
Created on Aug 13, 2021

Analyzes results of summarization for single items.

@author: immanueltrummer
'''
import argparse
import nminer.stats.common
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Analyze CP benchmark.')
    parser.add_argument('results_path', type=str, help='tsv file with results')
    args = parser.parse_args()
    
    df = pd.read_csv(args.results_path, sep='\t')
    df = nminer.stats.common.preprocess(df)
    print(df.info())

    plt.rcParams.update({'text.usetex': True, 'font.size':9,
                         'font.serif':['Computer Modern'],
                         'font.family':'serif'})
    plt.close('all')
    
    alg_groups = {'gen': ['r', 'R', 'G', 'V', 'P'], 'rl': ['B', 'P', 'SB', 'SP']}
    
    for scenario in range(4):
        scenario_data = df.query(f'scenario == {scenario}')
        path_prefix = f'plotsrev/S{scenario}_'
        
        for g_name, algs in alg_groups.items():
            file_path = path_prefix + f'{g_name}_qual.pdf'
            nminer.stats.common.perf_breakdown(
                scenario_data, 'approach', algs, 'Approach',
                'fquality', (-1.1, 1.1), 'Quality', 'linear', 
                file_path)
            
            file_path = path_prefix + f'{g_name}_time.pdf'
            nminer.stats.common.perf_breakdown(
                scenario_data, 'approach', algs, 'Approach', 
                'time', None, 'Time (s)', 'log', file_path)
    
    for g_name, algs in alg_groups.items():
        file_path = f'plotsrev/sx_{g_name}_time.pdf'
        nminer.stats.common.perf_plot(
            df, algs, 'time', None, 
            'Time (s)', 'log', file_path)
        file_path = f'plotsrev/sx_{g_name}_quality.pdf'
        nminer.stats.common.perf_plot(
            df, algs, 'fquality', (-1.1,1.1), 
            'Quality', 'linear', file_path)
    
    scenario_names = ['L', 'T', 'F', 'S']
    df['sc_name'] = df.apply(lambda r:scenario_names[r['scenario']], axis=1)
    
    df['relhits'] = df['chits'] / (df['chits'] + df['cmisses'])
    pro_hits = df.query('approach == "P"')
    hit_rate = pro_hits['relhits'].mean()
    print(f'Mean cache hit rate (proactive RL): {hit_rate}')
    nminer.stats.common.perf_breakdown(
        pro_hits, 'sc_name', scenario_names, 'Scenario',
        'relhits', (0, 1), 'Hit Ratio', 'linear', 'plotsrev/chits.pdf')
    
    df['relcost'] = df['nomergecost'] / (df['mergedcost'] + 0.001)
    sam_save = df.query('approach == "SB"')
    avg_save = sam_save['relcost'].mean()
    nminer.stats.common.perf_breakdown(
        sam_save, 'sc_name', scenario_names, 'scenario',
        'relcost', None, 'Savings', 'log', 'plotsrev/savings.pdf')
    print(f'Average relative savings: {avg_save}')