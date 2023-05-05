'''
Created on May 3, 2022

@author: immanueltrummer
'''
import argparse
import numpy as np
import pandas as pd
from statsmodels.stats import inter_rater


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('survey', type=str, help='Path to all survey responses')
    args = parser.parse_args()
    
    df = pd.read_csv(args.survey)
    a = df.iloc[:,8:12].dropna().apply(lambda r:r.argmax(), axis=1).values
    c = inter_rater.aggregate_raters(np.expand_dims(a, 0))[0]
    print(inter_rater.fleiss_kappa(c, method='unif'))