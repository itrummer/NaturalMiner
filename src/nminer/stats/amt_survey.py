'''
Created on May 25, 2022

@author: immanueltrummer
'''
import argparse
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import pandas as pd
import statistics
from statsmodels.stats import inter_rater

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to AMT result file')
    args = parser.parse_args()
    
    df = pd.read_csv(args.file)
    group_cols = ['Input.Vizbest', 'Answer.semantic-similarity.label']
    all_counts_by_cat = df.groupby(by=group_cols).size().unstack()
    all_counts_by_cat.fillna(0, inplace=True)
    
    v_votes = []
    p_votes = []
    labels = []
    for scenario_short in ['L', 'D', 'F', 'S']:
        labels += [f'{scenario_short}{i}' for i in range(1,6)]
    
    all_agreements = []
    for scenario in [
        'Among all laptops', 'Among all developers', 
        'Among all flights', 'Among all liquors']:
        print(f'Scenario: {scenario}')
        counts_by_cat = all_counts_by_cat.filter(like=scenario, axis=0)
        
        print(f'Counts: {counts_by_cat}')
        nr_subjects = counts_by_cat.shape[0]
        agreements = []
        for i in range(nr_subjects):
            counts = counts_by_cat.iloc[i]
            v_votes.append(counts.iloc[0])
            p_votes.append(counts.iloc[1])
            agreement = inter_rater.fleiss_kappa([counts], method='unif')
            agreements.append(agreement)
            all_agreements.append(agreement)
        print(f'Agreement: {statistics.mean(agreements)}')
    
    print(f'All agreement: {statistics.mean(all_agreements)}')
    print(v_votes)
    print(p_votes)
    print(labels)
    
    width = 0.35  # the width of the bars
    
    plt.rcParams.update({'text.usetex': True, 'font.size':9,
                         'font.serif':['Computer Modern'],
                         'font.family':'serif'})
    plt.close('all')
    
    figure, axes = plt.subplots(
        nrows=1, ncols=4, figsize=(3.25,1.5))

    for i, title in enumerate(['Laptops', 'Developers', 'Flights', 'Sales']):
        ax = axes[i]
        ax.set_title(title)
        ax.get_xaxis().set_visible(False)
        ax.yaxis.grid()
        if i == 0:
            ax.set_ylabel('Votes')
        ax.set_ylim(0,15)
        x = np.arange(5)
        rects1 = ax.bar(x - width/2, v_votes[i*5:i*5+5], width, label='V')
        rects2 = ax.bar(x + width/2, p_votes[i*5:i*5+5], width, label='P', hatch='///')
        print('Output')
        print(f'Vizier: {sum(v_votes[i*5:i*5+5])}')
        print(f'BABOONS: {sum(p_votes[i*5:i*5+5])}')
        
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Votes')
    
    # ax.set_title('Scores by group and gender')
    # ax.set_xticks(x, labels)
    # ax.legend()
    
    # ax.yaxis.grid()
    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    
    figure.tight_layout(pad=1.01)
    
    # plt.show()
    plt.savefig('amt.pdf')