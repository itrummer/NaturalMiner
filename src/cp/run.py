'''
Created on Jun 6, 2021

@author: immanueltrummer
'''
import cp.rl
import psycopg2
import stable_baselines3.common.env_checker
from stable_baselines3 import A2C, PPO
from cp.query import QueryEngine

with psycopg2.connect(
    database='picker', user='immanueltrummer') as connection:
    connection.autocommit = True
    #cmp_pred = "storename='Wilkie Liquors'"
    cmp_pred = "itemname='Kessler Blend Whiskey'"
    q_engine = QueryEngine('picker', 'immanueltrummer', '', 'liquor', cmp_pred)
    # env = cp.rl.PickingEnv(
        # connection, 'laptops', 
        # ['brand', 'processor_type', 'graphics_card', 'disk_space', 'ratings_5max'], 
        # ['display_size', 'discount_price', 'old_price'], 
        # "laptop_name='MateBook D Volta'", 1, 2, 2, 5,
        # 'Among all laptops', 
        # ['with <V> brand', 'with <V> CPU', 'with <V> graphics card', 
        # 'with <V> disk space', 'with <V> stars'],
        # ['its display size', 'its discounted price', 'its old price'])
    env = cp.rl.PickingEnv(
        connection, 'liquor', 
        ['city', 'countyname', 'categoryname', 'itemname'], 
        ['bottlessold', 'salevalue', 'volumesold'], 
        cmp_pred, 1, 2, 5, 20, 'Among all stores', 
        ['in <V>', 'in <V>', 'considering <V>'],
        [', the number of bottles per sale', 
         ', the dollar value per sale', 
         ', the volume per sale'],
        q_engine)

    stable_baselines3.common.env_checker.check_env(env)
    model = A2C(
        'MlpPolicy', env, verbose=True, 
        gamma=1.0, normalize_advantage=True)
    #model = PPO('MlpPolicy', env, gamma=1.0)
    model.learn(total_timesteps=1000)
    
    
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