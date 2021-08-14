'''
Created on Aug 13, 2021

@author: immanueltrummer
'''
import argparse
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

def perf_breakdown(
        df, approaches, metric, y_bounds, 
        y_label, y_mode, out_path):
    """ Plots a multi-plot for a subset of approaches.
    
    Args:
        df: data frame containing all benchmark results
        approaches: include those approaches in the plot
        metric: y axis summarizes this metric
        y_bounds: tuple of lower and upper y bound
        y_label: label of y axis
        y_mode: mode for y axis (e.g., linear or log)
        out_path: write plot to this file
    
    Returns:
        generated multi-plot
    """
    df = df.loc[:,['approach', 'nrfacts', 'nrpreds', metric]]
    df = df.query(f'approach in {approaches}')
        
    _, axes = plt.subplots(
        nrows=3, ncols=3, figsize=(3.5,3.5), 
        subplotpars=matplotlib.figure.SubplotParams(
            wspace=0.425, hspace=0.75))
    
    for x_pos in range(3):
        for y_pos in range(3):
            nr_facts = x_pos + 1
            nr_preds = y_pos + 1
            ax = axes[y_pos, x_pos]
            if y_bounds is not None:
                ax.set_ylim(y_bounds[0], y_bounds[1])
            ax.set_title(f'{nr_facts}F, {nr_preds}P')
            if x_pos == 0:
                ax.set_ylabel(y_label)
            if y_pos == 2:
                ax.set_xlabel('Approach')

            q = f'nrfacts == {nr_facts} and nrpreds == {nr_preds}'
            cur_df = df.query(q)
            plot_data = []
            for approach in approaches:
                app_data = cur_df.query(f'approach == "{approach}"')
                plot_data.append(app_data[metric])
                
            ax.boxplot(x=plot_data, labels=approaches, showmeans=True)
            ax.yaxis.grid()
    
    plt.yscale(y_mode)
    plt.tight_layout(1.05)
    plt.savefig(out_path)


def perf_plot(df, approaches, metric, y_bounds, y_label, out_path):
    """ Draws aggregate data per approach and scenario.
    
    Args:
        df: data frame containing data to draw
        approaches: draw data for these approaches
        metric: draw this metric on the y axis
        y_bounds: tuple with lower and upper y bounds
        y_label: axis label for y axis
        out_path: write output plot to this file
    """
    _, axes = plt.subplots(nrows=3, figsize=(3,3))
    scenario_names = ['Laptops (Size: 30 KB)', 
                      'Liquors (Size: 5 GB)', 
                      'Tools (Size: 10 MB)']
    
    for y_pos, scenario in enumerate([0, 2, 1]):        
        cur_df = df.query(f'scenario == {scenario}')
        df_pivot = cur_df.pivot(
            index=['testcase', 'nrfacts', 'nrpreds'], 
            columns='approach', values=metric)
        plot_data = [df_pivot[a] for a in approaches]
        
        ax = axes[y_pos]
        sc_name = scenario_names[scenario]
        ax.set_title(sc_name, fontsize=9)
        ax.set_ylabel(y_label)
        if y_bounds is not None:
            ax.set_ylim(y_bounds[0], y_bounds[1])
        ax.yaxis.grid()
        ax.boxplot(x=plot_data, labels=approaches, showmeans=True)
    
    plt.tight_layout(1.02)
    plt.subplots_adjust(hspace=0.7)
    plt.savefig(out_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Analyze CP benchmark.')
    parser.add_argument('results_path', type=str, help='tsv file with results')
    args = parser.parse_args()
    
    df = pd.read_csv(args.results_path, sep='\t')
    methods_short = {
        'rand1':'r', 'rand':'R', 'gen':'G', 'rlempty':'B',
        'sample':'S', 'rlcube':'C', 'rlproactive':'P'}
    df['approach'] = df['approach'].apply(lambda a:methods_short[a])
    df['fquality'] = df['bquality'].apply(lambda q:max(q,-1))
    print(df.info())

    plt.rcParams.update({'text.usetex': True, 'font.size':9})
    plt.close('all')
    
    alg_groups = {'gen': ['r', 'R', 'G', 'P'], 'rl': ['B', 'P', 'S']}
    
    for scenario in range(3):
        scenario_data = df.query(f'scenario == {scenario}')
        path_prefix = f'plots/S{scenario}_'
        
        for g_name, algs in alg_groups.items():
            file_path = path_prefix + f'{g_name}_qual.pdf'
            perf_breakdown(
                scenario_data, algs, 'fquality', 
                (-1.1, 1.1), 'Quality', 'linear', file_path)
            
            file_path = path_prefix + f'{g_name}_time.pdf'
            perf_breakdown(
                scenario_data, algs, 'time', 
                None, 'Time (s)', 'linear', file_path)
    
    for g_name, algs in alg_groups.items():
        file_path = f'plots/sx_{g_name}_time.pdf'
        perf_plot(df, algs, 'time', None, 'Time (s)', file_path)
        file_path = f'plots/sx_{g_name}_quality.pdf'
        perf_plot(df, algs, 'fquality', (-1.1,1.1), 'Quality', file_path)