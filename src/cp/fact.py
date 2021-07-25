'''
Created on Jul 21, 2021

@author: immanueltrummer
'''
from cp.pred import is_pred
import random

class Fact():
    """ Represents one fact in a data summary. """
    
    def __init__(self, nr_preds):
        """ Initialize fact for given number of predicates. 
        
        Args:
            nr_preds: (maximal) number of scope predicates
        """
        self.nr_preds = nr_preds
        self.nr_props = nr_preds + 1

    def change(self, prop_id, new_val):
        """ Change fact property to new value. 

        Args:
            prop_id: change value of this property
            new_val: assign property to this value
        """
        self.props[prop_id] = new_val
        
    def get_agg(self):
        """ Returns index of aggregate. """
        return self.props[self.nr_preds]
    
    def get_id(self):
        """ Returns ID string. """
        return "_".join([str(p) for p in self.props])
    
    def get_preds(self):
        """ Returns fact predicates (indices). """
        return self.props[0:self.nr_preds]
    
    def get_prop(self, prop_id):
        """ Returns current value for property. """
        return self.props[prop_id]
    
    def is_agg(self, prop_id):
        """ Returns true iff property represents aggregate. """
        return True if prop_id == self.nr_preds else False
    
    def random_init(self, pred_cnt, agg_cnt):
        """ Initializes fact randomly.
        
        Args:
            pred_cnt: number of possible predicates
            agg_cnt: number of possible aggregates
        """
        self.reset()
        for prop_id in range(self.nr_props):
            ub = agg_cnt if self.is_agg(prop_id) else pred_cnt
            self.props[prop_id] = random.randint(0, ub-1)
    
    def reset(self):
        """ Reset properties to default values. """
        self.props = [0] * self.nr_props

    
def fact_txt(fact, preamble, dim_cols, all_preds, 
             dims_tmp, agg_cols, q_engine, aggs_txt):
    """ Generate text describing fact. 
    
    Args:
        fact: a fact to describe
        preamble: start with this text
        dim_cols: list of dimension columns
        all_preds: list of all predicates
        dims_tmp: text templates for dimensions
        agg_cols: list of aggregation columns
        q_engine: used to calculate aggregates
        aggs_txt: text for aggregation columns
        
    Returns:
        text description of fact
    """
    f_parts = [preamble]
    preds = [all_preds[i] for i in fact.get_preds()]
    for pred in preds:
        if is_pred(pred):
            dim_idx = dim_cols.index(pred[0])
            dim_tmp = dims_tmp[dim_idx]
            dim_txt = dim_tmp.replace('<V>', str(pred[1]))
            f_parts.append(dim_txt)

    agg_idx = fact.get_agg()
    agg_col = agg_cols[agg_idx]
    rel_avg = q_engine.rel_avg(preds, agg_col)
    if rel_avg is None:
        return None

    percent = int(rel_avg * 100)
    percent_d = percent - 100
    if percent_d != 0:
        cmp_text = f'{abs(percent_d)}% '
        cmp_text += 'higher ' if percent_d > 0 else 'lower '
        cmp_text += 'than average'
    else:
        cmp_text = 'about average'                
    f_parts.append(f'{aggs_txt[agg_idx]} is {cmp_text}.')
    return ' '.join(f_parts).replace('_', ' ').replace('  ', ' ')