'''
Created on May 25, 2022

@author: immanueltrummer
'''
import argparse
import pandas as pd
import statistics
from statsmodels.stats import inter_rater

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to AMT result file')
    args = parser.parse_args()
    
    df = pd.read_csv(args.file)
    group_cols = ['HITId', 'Answer.semantic-similarity.label']
    counts_by_cat = df.groupby(by=group_cols).size().unstack()
    counts_by_cat.fillna(0, inplace=True)
    nr_subjects = counts_by_cat.shape[0]
    agreements = []
    for i in range(nr_subjects):
        counts = counts_by_cat.iloc[i]
        print(counts)
        agreement = inter_rater.fleiss_kappa([counts], method='unif')
        print(f'Agreement: {agreement}')
        agreements.append(agreement)
    print(agreements)
    print(statistics.mean(agreements))