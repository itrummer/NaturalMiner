'''
Created on May 4, 2022

@author: immanueltrummer
'''
import json
import os
import pandas as pd
import pathlib
import psycopg2.extras
import stable_baselines3
import streamlit as st
import sys
import time

cur_file_dir = os.path.dirname(__file__)
src_dir = pathlib.Path(cur_file_dir).parent.parent
sys.path.append(str(src_dir))
print(sys.path)

import nminer.algs.rl
import nminer.sql.pred
import nminer.text.sum

st.set_page_config(page_title='NaturalMiner')
st.markdown('''
NaturalMiner mines large data sets for patterns described in natural language.
''')

root_dir = src_dir.parent
scenarios_path = root_dir.joinpath('demo').joinpath('scenarios.json')
with open(scenarios_path) as file:
    scenarios = json.load(file)
nr_scenarios = len(scenarios)
print('Example scenarios loaded')

selected = st.selectbox(
    'Select Example (Optional)', options=range(nr_scenarios), 
    format_func=lambda idx:scenarios[idx]['scenario'])

with st.expander('Data Source'):
    connection_info = st.text_input(
        'Database connection details (format: "<Database>:<User>:<Password>"):',
        value=scenarios[selected]['dbconnection']).split(':')
    db_name = connection_info[0]
    db_user = connection_info[1]
    db_pwd = connection_info[2] if len(connection_info) > 1 else ''
    table = st.text_input(
        'Name of database table to analyze:', max_chars=100,
        value=scenarios[selected]['table'])

with st.expander('Text Templates'):
    preamble = st.text_input(
        'Preamble text for each fact:', 
        value=scenarios[selected]['preamble'])
    dims_info = st.text_area(
        'One dimension column per line ' +\
        '(format: "<Column>:<TemplateText>"; "<V>" as placeholder):', 
        value=scenarios[selected]['dimensions'], height=20).split('\n')
    aggs_info = st.text_area(
        'One aggregation column per line ' +\
        '(format: "<Column>:<TemplateText>"):', 
        value=scenarios[selected]['aggregates'], height=20).split('\n')
    nr_facts = st.slider(
        'Number of facts:', min_value=1, max_value=3,  
        value=scenarios[selected]['nr_facts'])
    nr_preds = st.slider(
        'Predicates per fact:', min_value=0, max_value=3, 
        value=scenarios[selected]['nr_preds'])

cmp_preds = st.text_area(
    'Data subsets to analyze (one SQL predicate per line):',
    value=scenarios[selected]['predicates'], height=20).split('\n')

label = st.text_input(
    'Pattern (natural language):', max_chars=100, 
    value=scenarios[selected]['goal'])

nr_iterations = st.slider(
    'Number of iterations:', 
    min_value=1, max_value=500, value=200)

print('Generated input elements')


if st.button('Find Pattern!'):
    print('Searching for pattern ...')
    
    result_cols = ['Predicate', 'Facts', 'Quality']
    result_df = pd.DataFrame(columns=result_cols, index=range(0))
    result_table = st.table(result_df)
    
    for cmp_pred in cmp_preds:
        st.write(f'Analyzing data satisfying predicate "{cmp_pred}" ...')
        dims_col_text = [d.split(':') for d in dims_info]
        aggs_col_text = [a.split(':') for a in aggs_info]
        t = {
            'table':table, 
            'dim_cols':[d[0] for d in dims_col_text],
            'agg_cols':[a[0] for a in aggs_col_text],
            'cmp_preds':[cmp_pred],
            'nr_facts':nr_facts, 'nr_preds':nr_preds, 
            'degree':5, 'max_steps':nr_iterations,
            'preamble':preamble, 
            'dims_tmp':[d[1] for d in dims_col_text],
            'aggs_txt':[a[1] for a in aggs_col_text]
        }
        with psycopg2.connect(
            database=db_name, user=db_user, 
            cursor_factory=psycopg2.extras.RealDictCursor) as connection:
            connection.autocommit = True
            
            start_s = time.time()
            all_preds = nminer.sql.pred.all_preds(
                connection, t['table'], 
                t['dim_cols'], cmp_pred)
            sum_eval = nminer.text.sum.SumEvaluator(
                1, 'facebook/bart-large-mnli', label)
            env = nminer.algs.rl.PickingEnv(
                connection, **t, all_preds=all_preds,
                c_type='proactive', cluster=True,
                sum_eval=sum_eval)
            model = stable_baselines3.A2C(
                'MlpPolicy', env, verbose=True, 
                gamma=1.0, normalize_advantage=True)
            model.learn(total_timesteps=nr_iterations)
            total_s = time.time() - start_s
            rated_sums = env.s_eval.text_to_reward
            actual_sums = [i for i in rated_sums.items() if i[0] is not None]
            sorted_sums = sorted(actual_sums, key=lambda s: s[1])
            if sorted_sums:
                b_sum = sorted_sums[-1]
            else:
                b_sum = ('(No valid summary generated)', -10)
            
            new_row = pd.DataFrame([
                [cmp_pred, b_sum[0], b_sum[1]]], 
                columns=result_cols)
            result_table.add_rows(new_row)
            
    st.write('All data subsets generated!')