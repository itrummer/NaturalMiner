'''
Created on Jun 6, 2021

@author: immanueltrummer
'''
import cp.rl
import psycopg2
import stable_baselines3.common.env_checker

with psycopg2.connect(
    database='picker', user='immanueltrummer') as connection:
    env = cp.rl.PickingEnv(connection, 'laptops', ['price'], ['disk_space'], 
                           "model='MacBook Air'", 2, 2, 1)
    stable_baselines3.common.env_checker.check_env(env)