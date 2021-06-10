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
        "laptop_name='MateBook D Volta'", 2, 2, 2)
    stable_baselines3.common.env_checker.check_env(env)
    model = A2C(
        'MlpPolicy', env, verbose=True,
        normalize_advantage=True)
    model.learn(total_timesteps=100)
    
    best_summary = env.best_summary()
    print(f'Best summary: "{best_summary}"')