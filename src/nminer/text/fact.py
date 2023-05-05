'''
Created on Jul 21, 2021

@author: immanueltrummer
'''
from cp.sql.pred import is_pred
import math
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
        self.reset()

    @staticmethod
    def from_props(props):
        """ Initializes new fact from properties.
        
        Args:
            props: fact properties
            
        Returns:
            new fact with given properties
        """
        nr_preds = len(props) - 1
        fact = Fact(nr_preds)
        for p_id, val in enumerate(props):
            fact.change(p_id, val)
        
        return fact

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
        for prop_id in range(self.nr_props):
            ub = agg_cnt if self.is_agg(prop_id) else pred_cnt
            self.props[prop_id] = random.randint(0, ub-1)
    
    def reset(self):
        """ Reset properties to default values. """
        self.props = [0] * self.nr_props
    
    def set_agg(self, value):
        """ Set aggregate to given value.
        
        Args:
            value: index of aggregation column
        """
        self.props[self.nr_preds] = value
    
    def set_pred(self, pred_idx, value):
        """ Set predicate to given value.
        
        Args:
            pred_idx: index of predicate to set
            value: set predicate to this value
        """
        self.props[pred_idx] = value
    
    def to_txt(self, preamble, dim_cols, all_preds, 
             dims_tmp, agg_cols, q_engine, aggs_txt):
        """ Generate text describing fact with confidence. 
        
        Args:
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
        preds = [all_preds[i] for i in self.get_preds()]
        preds = list(filter(lambda p:is_pred(p), preds))
        for pred in preds:
            dim_idx = dim_cols.index(pred[0])
            dim_tmp = dims_tmp[dim_idx]
            dim_txt = dim_tmp.replace('<V>', str(pred[1]))
            f_parts.append(dim_txt)
    
        agg_idx = self.get_agg()
        agg_col = agg_cols[agg_idx]
        rel_avg, row_cnt = q_engine.rel_avg(preds, agg_col)
        if rel_avg is None:
            return None, None
    
        percent = round(rel_avg * 100)
        percent_d = percent - 100
        if percent_d != 0:
            cmp_text = f'{abs(percent_d)}% '
            cmp_text += 'higher ' if percent_d > 0 else 'lower '
            cmp_text += 'than average'
        else:
            cmp_text = 'about average'                
        f_parts.append(f'{aggs_txt[agg_idx]} is {cmp_text}.')
        text = ' '.join(f_parts).replace('_', ' ').replace('  ', ' ')
        
        return text, self._confidence(row_cnt)
    
    def _confidence(self, row_cnt):
        """ Calculates confidence that text is accurate.
        
        For this implementation, we assume that all digits
        are used when generating text. Otherwise, we
        would need to consider the actual percentage.
        
        Args:
            row_cnt: rows considered for entity average

        Returns:
            confidence that text is accurate
        """
        return 2 * math.exp(-2*0.005**2/(row_cnt*1))