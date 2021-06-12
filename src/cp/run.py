'''
Created on Jun 6, 2021

@author: immanueltrummer
'''
import cp.rl
import psycopg2
import stable_baselines3.common.env_checker
from stable_baselines3 import A2C

with psycopg2.connect(
    database='picker', user='immanueltrummer') as connection:
    env = cp.rl.PickingEnv(
        connection, 'laptops', 
        ['brand', 'processor_type', 'graphics_card', 'disk_space', 'ratings_5max'], 
        ['display_size', 'discount_price', 'old_price'], 
        "laptop_name='MateBook D Volta'", 1, 2, 2)
    stable_baselines3.common.env_checker.check_env(env)
    model = A2C(
        'MlpPolicy', env, verbose=True, 
        gamma=1.0, normalize_advantage=True)
    model.learn(total_timesteps=1000)
    
    best_summary = env.best_summary()
    print(f'Best summary: "{best_summary}"')
    for best in [True, False]:
        print(f'Top-K summaries (Best: {best})')
        topk_summaries = env.topk_summaries(2, best)
        for s in topk_summaries:
            print(s)