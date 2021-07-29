'''
Created on Jul 21, 2021

@author: immanueltrummer
'''
import cp.fact
import cp.query
import cp.sum
import time
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import GPT2Tokenizer
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer

# Prepare extractive summarization
mnli_tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')
mnli_model = AutoModelForSequenceClassification.from_pretrained(
    'roberta-large-mnli')
mnli_model.eval()


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
    stats.update(q_engine.statistics())
    stats.update(s_gen.statistics())
    stats.update(s_eval.statistics())
    
    return text_to_quality, stats


def escape_text(raw_text):
    """ Escape generated text.
    
    Args:
        raw_text: text potentially containing double quotes
        
    Returns:
        text with escaped double quotes
    """
    return raw_text.replace('"', '""')


def gen_rl(timeout_s, **kwargs):
    """ Generative baseline using reinforcement learning. 
    
    Args:
        timeout_s: timeout in seconds
        kwargs: dictionary with arguments
        
    Returns:
        Dictionary mapping summaries to quality values
    """
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_model = GPT2HeadWithValueModel.from_pretrained('gpt2')
    gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained('gpt2')
    
    ppo_config = {'batch_size': 1, 'forward_batch_size': 1}
    ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, **ppo_config)
    
    r_params = kwargs.copy()
    r_params['nr_facts'] = 10
    r_params['nr_preds'] = 1
    s_eval = cp.sum.SumEvaluator()
    start_s = time.time()
    t_to_q = {}
    eval_total_s = 0
    
    while True:
        rt_to_q, _ = rand_sums(1, float('inf'), **r_params)
        
        rand_facts = list(rt_to_q.keys())[0]
        if rand_facts is not None:
            prompt_pt = gpt2_tokenizer.encode(rand_facts, return_tensors="pt")        
            sum_pt = respond_to_batch(gpt2_model, prompt_pt)
            summary = gpt2_tokenizer.decode(sum_pt[0,:])
            print(f'Summary: {summary}')
            
            eval_start_s = 0
            with torch.no_grad():
                pos_reward = s_eval.evaluate(summary)
                
                mnli_input = rand_facts + '</s>' + summary
                mnli_e = mnli_tokenizer(
                    mnli_input, return_tensors='pt', 
                    truncation=True)
                mnli_pred = torch.softmax(
                    mnli_model(**mnli_e).logits, 
                    dim=1).tolist()[0]
                mnli_reward = mnli_pred[2]
                
                reward = pos_reward + mnli_reward
                reward_pt = torch.tensor([reward])
                
            ppo_trainer.step(prompt_pt, sum_pt, reward_pt)
            esc_summary = '"' + escape_text(summary) + '"'
            t_to_q[esc_summary] = pos_reward
            eval_total_s += time.time() - eval_start_s
            
        else:
            t_to_q['None'] = -10
        
        total_s = time.time() - start_s
        if total_s > timeout_s:
            break
    
    p_stats = {}
    p_stats['time'] = total_s
    p_stats['evaluation_time'] = eval_total_s
    p_stats['cache_hits'] = -1
    p_stats['cache_misses'] = -1
    
    return t_to_q, p_stats