'''
Created on May 4, 2022

@author: immanueltrummer
'''
import json
import os
import pathlib
import streamlit as st
import sys

cur_file_dir = os.path.dirname(__file__)
src_dir = pathlib.Path(cur_file_dir).parent.parent
sys.path.append(src_dir)

st.set_page_config(page_title='BABOONS')
st.markdown('''
BABOONS (Black-Box Optimization of Natural Language Summaries) generates
comparative summaries from relational tables that are optimized 
according to user-defined criteria.
''')

root_dir = src_dir.parent
scenarios_path = root_dir.joinpath('demo').joinpath('scenarios.json')
with open(scenarios_path) as file:
    scenarios = json.load(file)
nr_scenarios = len(scenarios)

selected = st.selectbox(
    'Select Example (Optional)', options=range(nr_scenarios), 
    format_func=lambda idx:scenarios[idx]['scenario'])

connection = st.text_input(
    'Database connection details (format: "<Database>:<User>:<Password>"):',
    value=scenarios[selected]['dbconnection'])
table = st.text_input(
    'Name of database table to summarize:', max_chars=100,
    value=scenarios[selected]['table'])
items = st.text_input(
    'Data subsets to describe (i.e., an SQL predicate):',
    value=scenarios[selected][''])
preamble = st.text_input('Preamble text for each fact:')
dims = st.text_area(
    'One dimension column per line ' +\
    '(format: "<Column>:<TemplateText>"; "<V>" as placeholder):', 
    height=20)
aggs = st.text_area(
    'One aggregation column per line (format: "<Column>:<TemplateText>"):', 
    height=20)
label = st.text_input(
    'Communication goal:', max_chars=100)
nr_facts = st.slider(
    'Number of facts:', min_value=1, max_value=3, value=1)
nr_preds = st.slider(
    'Maximal number of predicates per fact:', 
    min_value=0, max_value=3, value=2)
nr_iterations = st.slider(
    'Number of optimizer iterations:', 
    min_value=1, max_value=500, value=200)

if st.button('Generate Summary!'):
    st.write('The summary is generated!')