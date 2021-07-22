'''
Created on Jun 6, 2021

@author: immanueltrummer
'''
import cp.rl
import psycopg2
import stable_baselines3.common.env_checker
import time
from stable_baselines3 import A2C, PPO
from cp.query import QueryEngine

def run_main(test_case):
    """ Benchmarks primary method on test case.
    
    Args:
        test_case: describes test case
        
    Returns:
        best, run time, pure processing, quality
    """
    pass

with psycopg2.connect(
    database='picker', user='immanueltrummer') as connection:
    connection.autocommit = True
    
    # From URL: https://www.kaggle.com/tylerx/melbourne-airbnb-open-data?select=listings_summary_dec18.csv
    cmp_pred = 'id = 21610859'
    q_engine = QueryEngine('picker', 'immanueltrummer', '', 'melbournedec18', cmp_pred)
    env = cp.rl.PickingEnv(
        connection=connection, table='melbournedec18', 
        dim_cols=['host_name', 'neighbourhood', 'neighbourhood_group', 'room_type'],
        agg_cols=['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                  'calculated_host_listings_count', 'availability_365'],
        cmp_pred=cmp_pred, nr_facts=1, nr_preds=2, degree=5, max_steps=20,
        preamble='Among all apartments, ', 
        dims_tmp=['hosted by <V>', 'in <V>', 'in <V>', 'of type <V>'],
        aggs_txt=[', the price', ', the minimum stay', ', the number of reviews',
                  ', the reviews per month', ', the number of listings by the same host',
                  ', the number of available days over the year'],
        q_engine=q_engine)

    stable_baselines3.common.env_checker.check_env(env)
    model = A2C(
        'MlpPolicy', env, verbose=True, 
        gamma=1.0, normalize_advantage=True)
    #model = PPO('MlpPolicy', env, gamma=1.0)
    
    start_s = time.time()
    model.learn(total_timesteps=10000)
    total_s = time.time() - start_s
    print(f'Optimization took {total_s} seconds')
    
    best_summary = env.best_summary()
    print(f'Best summary: "{best_summary}"')
    for best in [True, False]:
        print(f'Top-K summaries (Best: {best})')
        topk_summaries = env.topk_summaries(2, best)
        for s in topk_summaries:
            print(s)
            
    print('Facts generated:')
    for f in env.fact_to_text.values():
        print(f'{f}')