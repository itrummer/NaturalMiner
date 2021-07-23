'''
Created on Jul 21, 2021

@author: immanueltrummer
'''
import cp.fact
import cp.query
import cp.sum
import time

def rand_sums(
        nr_sums, timeout_s, connection, table, dim_cols, 
        agg_cols, cmp_pred, nr_facts, nr_preds, preamble, 
        dims_tmp, aggs_txt, all_preds, **kwargs):
    """ Generates and evaluates one random summary.
    
    Args:
        nr_sums: max. number of summaries to generate
        timeout_s: max. time in seconds
        connection: connects to database
        table: summarize this table
        dim_cols: dimension columns
        agg_cols: aggregation columns
        cmp_pred: identifies entity to advertise
        nr_facts: max. number of facts in summary
        nr_preds: max. number of predicates per fact
        preamble: start summary with this text
        dims_tmp: text templates for dimensions
        aggs_txt: text describing aggregates
        all_preds: all possible predicates
    
    Returns:
        Dictionary mapping summaries to reward, statistics
    """
    start_s = time.time()
    q_engine = cp.query.QueryEngine(
        connection, table, cmp_pred, float('inf'))
    s_gen = cp.sum.SumGenerator(
        all_preds, preamble, dim_cols, dims_tmp, 
        agg_cols, aggs_txt, q_engine)
    s_eval = cp.sum.SumEvaluator()

    pred_cnt = len(all_preds)
    agg_cnt = len(agg_cols)
    text_to_quality = {}
    
    counter = 0
    while counter < nr_sums:
        counter += 1
        
        facts = []
        for _ in range(nr_facts):
            fact = cp.fact.Fact(nr_preds)
            fact.random_init(pred_cnt=pred_cnt, agg_cnt=agg_cnt)
            facts.append(fact)
    
        text = s_gen.generate(facts)
        quality = s_eval.evaluate(text)
        text_to_quality[text] = quality
        
        total_s = time.time() - start_s
        if total_s > timeout_s:
            break
    
    stats = {'time':total_s}
    stats.update(s_gen.statistics())
    stats.update(s_eval.statistics())
    
    return text_to_quality, stats