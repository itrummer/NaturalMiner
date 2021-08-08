'''
Created on Jul 22, 2021

@author: immanueltrummer
'''
import cp.text.fact
import logging
import time
import transformers

class SumGenerator():
    """ Class for generating summaries. """
    
    def __init__(self, all_preds, preamble, dim_cols, 
                 dims_tmp, agg_cols, aggs_txt, q_engine):
        """ Initializes parameters and fact cache. 
        
        Args:
            all_preds: all relevant predicates
            preamble: starts each fact text
            dim_cols: dimension columns
            dims_tmp: dimension templates
            agg_cols: aggregation columns
            aggs_txt: text describing column
            q_engine: used for querying
        """
        self.all_preds = all_preds
        self.preamble = preamble
        self.dim_cols = dim_cols
        self.dims_tmp = dims_tmp
        self.agg_cols = agg_cols
        self.aggs_txt = aggs_txt
        self.q_engine = q_engine
        self.fact_to_text = {}
        self.gen_s = 0
    
    def generate(self, facts):
        """ Generate text describing given facts. 
        
        Args:
            facts: describe those facts
            
        Returns:
            text string describing facts
        """
        start_s = time.time()
        s_parts = []
        for fact in facts:
            f_id = fact.get_id()
            if f_id in self.fact_to_text:
                f_txt = self.fact_to_text[f_id]
            else:
                f_txt = cp.text.fact.fact_txt(
                    fact, preamble=self.preamble, dim_cols=self.dim_cols, 
                    all_preds=self.all_preds, dims_tmp=self.dims_tmp, 
                    agg_cols=self.agg_cols, q_engine=self.q_engine, 
                    aggs_txt=self.aggs_txt)
                self.fact_to_text[f_id] = f_txt            
            if f_txt is None:
                return None
            s_parts.append(f_txt)
    
        self.gen_s += time.time() - start_s
        return ' '.join(s_parts)
    
    def statistics(self):
        """ Returns performance statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {'generation_time':self.gen_s}

class SumEvaluator():
    """ Class for evaluating data summaries. """
    
    judge = transformers.pipeline(
        "sentiment-analysis", 
        model="siebert/sentiment-roberta-large-english")
    
    def __init__(self):
        """ Initializes cache, statistics, and model. """
        self.eval_s = 0
        self.text_to_reward = {}
    
    def evaluate(self, text):
        """ Evaluate quality of current summary. 
        
        Args:
            text: summary text to evaluate
            
        Returns:
            numerical value (higher is more positive)
        """
        start_s = time.time()
        if text is None:
            reward = -10
        else:
            if text in self.text_to_reward:
                reward = self.text_to_reward[text]
            else:
                sent = self.judge(text)[0]
                label = sent['label']
                score = sent['score']
                if label == 'POSITIVE':
                    reward = score
                else:
                    reward = -score
                self.text_to_reward[text] = reward
    
            logging.debug(f'Reward {reward} for "{text}"')
        
        self.eval_s += time.time() - start_s
        return reward
    
    def statistics(self):
        """ Returns performance statistics. 
        
        Returns:
            dictionary with performance statistics
        """
        return {'evaluation_time':self.eval_s}