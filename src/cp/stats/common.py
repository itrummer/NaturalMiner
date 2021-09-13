'''
Created on Aug 21, 2021

@author: immanueltrummer
'''
import math
import matplotlib
import matplotlib.pyplot as plt

def perf_breakdown(
        df, x_dim, x_vals, x_label, metric, 
        y_bounds, y_label, y_mode, out_path):
    """ Plots a multi-plot for a subset of approaches.
    
    Args:
        df: data frame containing all benchmark results
        x_dim: use this column for the x-axis
        x_vals: consider those values for x-axis
        x_label: use this label for x-axis
        metric: y axis summarizes this metric
        y_bounds: tuple of lower and upper y bound
        y_label: label of y axis
        y_mode: mode for y axis (e.g., linear or log)
        out_path: write plot to this file
    
    Returns:
        generated multi-plot
    """
    df = df.loc[:,[x_dim, 'nrfacts', 'nrpreds', metric]]
    df = df.query(f'{x_dim} in {x_vals}')
        
    _, axes = plt.subplots(
        nrows=3, ncols=3, figsize=(3.35,3), # 3.5, 3.5 
        subplotpars=matplotlib.figure.SubplotParams(
            wspace=0, hspace=0.75))
    
    for x_pos in range(3):
        for y_pos in range(3):
            nr_facts = x_pos + 1
            nr_preds = y_pos + 1
            ax = axes[y_pos, x_pos]
            if y_bounds is not None:
                ax.set_ylim(y_bounds[0], y_bounds[1])
            ax.set_title(f'{nr_facts}F, {nr_preds}P')
            if y_pos == 2:
                ax.set_xlabel(x_label)
            if x_pos == 0:
                ax.set_ylabel(y_label)

            q = f'nrfacts == {nr_facts} and nrpreds == {nr_preds}'
            cur_df = df.query(q)
            plot_data = []
            for approach in x_vals:
                app_data = cur_df.query(f'{x_dim} == "{approach}"')
                plot_data.append(app_data[metric])
                
            ax.boxplot(x=plot_data, labels=x_vals, showmeans=True)
            ax.yaxis.grid()
            ax.set_yscale(y_mode)

    #plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.tight_layout(1.05)
    plt.savefig(out_path)


def perf_plot(df, approaches, metric, y_bounds, y_label, y_mode, out_path):
    """ Draws aggregate data per approach and scenario.
    
    Args:
        df: data frame containing data to draw
        approaches: draw data for these approaches
        metric: draw this metric on the y axis
        y_bounds: tuple with lower and upper y bounds
        y_label: axis label for y axis
        y_mode: type of y axis (linear vs. log)
        out_path: write output plot to this file
    """
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(3,2.5))
    scenario_names = ['Laptops (30 KB)', 
                      'Tools (10 MB)',
                      'Flights (900 MB)',
                      'Liquors (5 GB)']
    for scenario_id, scenario in enumerate([0, 1, 2, 3]):
        cur_df = df.query(f'scenario == {scenario}')
        df_pivot = cur_df.pivot(
            index=['testcase', 'nrfacts', 'nrpreds'], 
            columns='approach', values=metric)
        plot_data = [df_pivot[a] for a in approaches]
        
        x_pos = math.floor(scenario_id / 2)
        y_pos = math.floor(scenario_id % 2)
        ax = axes[x_pos][y_pos]
        sc_name = scenario_names[scenario]
        ax.set_title(sc_name, fontsize=9)
        ax.set_ylabel(y_label)
        if y_bounds is not None:
            ax.set_ylim(y_bounds[0], y_bounds[1])
        ax.yaxis.grid()
        ax.set_yscale(y_mode)
        ax.boxplot(x=plot_data, labels=approaches, showmeans=True)
    
    plt.tight_layout(1.05)
    #plt.subplots_adjust(hspace=0.7)
    plt.savefig(out_path)

def preprocess(df):
    """ Pre-process data frame containing CP results. 
    
    Args:
        df: data frame with CP results
    
    Returns:
        results with normalized approach and quality values
    """
    methods_short = {
        'rand1':'r', 'rand':'R', 'gen':'G', 'rlempty':'B',
        'sampleempty':'SB', 'sampleproactive':'SP', 
        'rlcube':'C', 'rlproactive':'P', 'rlNCproactive':'PN',
        'sample':'SP', 'simple':'SC', 'cluster':'CBR', 
        'bestof':'CP'}
    df['approach'] = df['approach'].apply(lambda a:methods_short[a])
    df['fquality'] = df['bquality'].apply(lambda q:max(q,-1))
    return df