'''
Created on May 5, 2023

@author: immanueltrummer
'''
import psycopg2.extras
import stable_baselines3
import time
import nminer.algs.rl
import nminer.sql.pred
import nminer.text.sum


def mine(
        db_name, db_user, db_pwd, table,
        preamble, dim_cols, dims_txt, agg_cols, aggs_txt, 
        target, nl_pattern, hg_model='facebook/bart-large-mnli',
        nr_facts=1, nr_preds=2, degree=5, nr_iterations=200):
    """
    Mines for facts that relate to natural language pattern.
    
    Args:
        db_name: name of Postgres database
        db_user: name of Postgres database user
        db_pwd: password of Postgres database user
        table: mine data in this table
        preamble: all facts start with this text
        dim_cols: names of categorical table columns for equality predicates
        dims_txt: list of text templates for predicates on dimension columns
        agg_cols: names of numerical table columns for aggregates
        aggs_txt: list of text templates for reporting aggregates
        target: SQL predicate describing target data compared to full data
        nl_pattern: search for facts that semantically relate to this pattern
        hg_model: ID of Huggingface Transformer model
        nr_facts: mine for relevant combinations of up to that many facts
        nr_preds: mine for facts using up to that many predicates
        degree: each search space node has that many neighbors
        nr_iterations: evaluate that many fact combinations
    
    Returns:
        a pair: a text reporting facts and a relevance score (w.r.t. pattern)
    """
    with psycopg2.connect(
        database=db_name, user=db_user, password=db_pwd,
        cursor_factory=psycopg2.extras.RealDictCursor) as connection:
        connection.autocommit = True
        
        # Mine table for facts related to pattern
        start_s = time.time()
        all_preds = nminer.sql.pred.all_preds(
            connection, table, dim_cols, target)
        sum_eval = nminer.text.sum.SumEvaluator(
            1, hg_model, nl_pattern)
        env = nminer.algs.rl.PickingEnv(
            connection, table, dim_cols, 
            agg_cols, [target], nr_facts, nr_preds,
            degree, nr_iterations, preamble, dims_txt, 
            aggs_txt, all_preds, c_type='proactive', 
            cluster=True, sum_eval=sum_eval)
        model = stable_baselines3.A2C(
            'MlpPolicy', env, verbose=True, 
            gamma=1.0, normalize_advantage=True)
        model.learn(total_timesteps=nr_iterations)
        total_s = time.time() - start_s
        print(f'Mining time: {total_s} seconds')
        
        # Find facts with highest relevance score
        rated_sums = env.s_eval.text_to_reward
        actual_sums = [i for i in rated_sums.items() if i[0] is not None]
        sorted_sums = sorted(actual_sums, key=lambda s: s[1])
        if sorted_sums:
            b_sum = sorted_sums[-1]
        else:
            b_sum = (None, -10)
    
        print(b_sum)
        return b_sum