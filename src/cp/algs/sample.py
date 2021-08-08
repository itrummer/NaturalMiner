'''
Created on Aug 7, 2021

@author: immanueltrummer
'''
from cp.sql.query import AggQuery, GroupQuery
import copy
import cp.cache.static
import cp.text.fact
import cp.algs.rl
import cp.text.sum
import logging
from stable_baselines3 import A2C
import time

def learn_on_sample(connection, test_case, all_preds):
    """ Learn good summaries for a data sample.
    
    Args:
        connection: connection to database system
        test_case: summarizes test case
        all_preds: all available predicates
    
    Returns:
        summaries sorted by estimated quality (descending), statistics
    """
    sample_case = test_case.copy()
    full_table = test_case['table']
    table_sample = f'(select * from {full_table} limit 10000) as S'
    sample_case['table'] = table_sample

    cache = cp.cache.static.EmptyCache()    
    env = cp.algs.rl.PickingEnv(
        connection, **sample_case, all_preds=all_preds,
        cache=cache, proactive=False)
    model = A2C(
        'MlpPolicy', env, verbose=True, 
        gamma=1.0, normalize_advantage=True)
    model.learn(total_timesteps=200)

    s_sums = sorted(
        env.props_to_rewards.items(),
        key=lambda s: s[1], reversed=True)
    return [
        [cp.text.fact.Fact.from_props(p) for p in s[0]] 
        for s in s_sums], env.statistics()

def size_delta(g_query, a_query):
    

def fill_cache(table, all_preds, cmp_pred, agg_cols, sam_sums, cache):
    g_queries = []
    for fact in sam_sums[0]:
        a_q = AggQuery.from_fact(table, all_preds, cmp_pred, agg_cols, fact)
        g_q = GroupQuery.from_query(a_q)
        g_queries += [g_q]
    
    for sum in sam_sums:
        for fact in sum:
            a_q = AggQuery.from_fact(table, all_preds, cmp_pred, agg_cols, fact)
            mergeable = [g for g in g_queries if g.can_merge(a_q)]
            min_g = min([])
            for g in g_queries:
                

def full_data_summaries(connection, test_case, all_preds, sam_sums):
    """ Generates summaries by considering the full data set.
    
    Args:
        connection: connection to database
        test_case: test case to solve
        all_preds: all possible predicates
        sam_sums: summaries selected via sample
    
    Returns:
        a subset of summaries refering to full data set
    """
    table = test_case['table']
    cmp_pred = test_case['cmp_pred']
    dim_cols = test_case['dim_cols']
    agg_cols = test_case['agg_cols']
    preamble = test_case['preamble']
    dims_tmp = test_case['dims_tmp']
    aggs_txt = test_case['aggs_txt']
    text_to_reward = {}
    
    cache = cp.cache.proactive.DynamicCache(connection, table, cmp_pred)
    is sam_sums:
        fill_cache(cache)
    
    q_engine = cp.sql.query.QueryEngine(connection, table, cmp_pred, cache)
    s_gen = cp.text.sum.SumGenerator(
        all_preds, preamble, dim_cols, 
        dims_tmp, agg_cols, aggs_txt, 
        q_engine)
    s_eval = cp.text.sum.SumEvaluator()
    
    nr_sums = min(len(sam_sums), 5)
    for facts in sam_sums[0:nr_sums]:
        
        queries = []
        for fact in facts:
            query = AggQuery.from_fact(
                table, all_preds, cmp_pred, 
                agg_cols, fact)
            queries.append(query)
        
        if not [q for q in queries if not cache.can_answer(q)]:
            text = s_gen.generate(facts)
            reward = s_eval.evaluate(text)
            text_to_reward[text] = reward
    
    return text_to_reward

def run_sampling(connection, test_case, all_preds):
    """ Run sampling algorithm. 
    
    Args:
        connection: connection to database
        test_case: summarize for this test case
        all_preds: all available predicates
        
    Returns:
        summaries with quality, performance statistics
    """
    start_s = time.time()
    s_sums, p_stats = learn_on_sample(connection, test_case, all_preds)
    f_to_r = full_data_summaries(connection, test_case, all_preds, s_sums)
    f_sums = [f[0] for f in f_to_r.items()]
    total_s = time.time() - start_s
    p_stats['time'] = total_s
    logging.debug(f'Optimization took {total_s} seconds')
    
    return f_sums, p_stats