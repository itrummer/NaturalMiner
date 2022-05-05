'''
Created on Jul 22, 2021

@author: immanueltrummer
'''
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
        self.fact_to_conf = {}
        self.gen_s = 0
    
    def generate(self, facts):
        """ Generate text describing given facts. 
        
        Args:
            facts: describe those facts
            
        Returns:
            text string describing facts, confidence
        """
        start_s = time.time()
        s_parts = []
        s_conf = 1
        for fact in facts:
            f_id = fact.get_id()
            if f_id in self.fact_to_text:
                f_txt = self.fact_to_text[f_id]
                f_conf = self.fact_to_conf[f_id]
            else:
                f_txt, f_conf = fact.to_txt(
                    preamble=self.preamble, dim_cols=self.dim_cols, 
                    all_preds=self.all_preds, dims_tmp=self.dims_tmp, 
                    agg_cols=self.agg_cols, q_engine=self.q_engine, 
                    aggs_txt=self.aggs_txt)
                self.fact_to_text[f_id] = f_txt
                self.fact_to_conf[f_id] = f_conf
                
            if f_txt is None:
                return None, None
            s_parts.append(f_txt)
            s_conf *= f_conf
    
        self.gen_s += time.time() - start_s
        return ' '.join(s_parts), s_conf
    
    def statistics(self):
        """ Returns performance statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {'generation_time':self.gen_s}


class SumEvaluator():
    """ Class for evaluating data summaries. """
    
    def __init__(self, goal=0, 
                 model_name='siebert/sentiment-roberta-large-english', 
                 label=''):
        """ Initializes cache, statistics, and model. 
        
        Args:
            goal: optimization goal (0 for default, 1 for customized)
            model_name: name of Huggingface model used for text evaluation
            label: compare text to this label for custom goals
        """
        self.goal = goal
        self.model_name = model_name
        self.label = label
        self.model = self._init_model()
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
                reward = self._evaluate_text(text)
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
    
    def _init_model(self):
        """ Initialize model used for text evaluation.
        
        Returns:
            newly initialized model for evaluation
        """
        if self.goal == 0:
            task = 'sentiment-analysis'
        elif self.goal == 1:
            task = 'zero-shot-classification'
        else:
            raise ValueError(f'Error - unknown goal code: {self.goal}')
        return transformers.pipeline(task, model=self.model_name)
    
    def _evaluate_text(self, text):
        """ Evaluate quality of input text. 
        
        Returns:
            quality value between -1 (or 0) and +1 (higher is better)
        """
        if self.goal == 0:
            sent = self.model(text)[0]
            label = sent['label']
            score = sent['score']
            if label == 'POSITIVE':
                return score
            else:
                return -score

        elif self.goal == 1:
            result = self.model(text, [self.label])
            return result['scores'][0]

        else:
            raise ValueError(f'Error - unknown goal code: {self.goal}')