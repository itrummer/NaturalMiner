'''
Created on May 28, 2022

@author: immanueltrummer
'''
import argparse
import pandas as pd
import statistics
from statsmodels.stats import inter_rater

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to input file')
    args = parser.parse_args()
    
    df = pd.read_csv(args.path)
    print(f'DF columns: {df.columns}')
    
    
    nr_answers = df.groupby(by=['HITId']).size()
    print(nr_answers)
    print(f'Min answers : {min(nr_answers)}')
    print(f'Max answers: {max(nr_answers)}')
    
    groups = df.groupby(by=['HITId','Answer.semantic-similarity.label'])
    yn = groups.size().unstack()
    nr_topics = yn.shape[0]
    print(f'Nr topics: {nr_topics}')
    
    nr_yes = sum(yn.iloc[:,0] > yn.iloc[:,1])
    print(f'Nr. yes answers: {nr_yes}')
    
    agreements = []
    for i in range(nr_topics):
        y = yn.iloc[i,0]
        n = yn.iloc[i,1]
        agreement = inter_rater.fleiss_kappa([[y,n]], 'unif')
        agreements.append(agreement)
        
    mean_agreement = statistics.mean(agreements)
    print(f'Inter-rater agreement: {mean_agreement}')