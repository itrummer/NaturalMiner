'''
Created on Jul 31, 2021

@author: immanueltrummer
'''
from abc import ABC, abstractmethod
from cp.pred import is_pred
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class AggQuery():
    """ Represents a simple aggregation query. """
    
    table: str
    eq_preds: List[Tuple[str, str]]
    cmp_pred: str
    agg_col: str
        
    def pred_cols(self):
        """ Returns columns of query predicates. 
        
        Returns:
            frozen set with predicate columns
        """
        return frozenset([p[0] for p in self.eq_preds if is_pred(p)])


class AggCache(ABC):
    """ Common superclass for all cache types. """
    
    @abstractmethod
    def can_answer(self, query):
        """ Checks if cache contains answer to query.
        
        Args:
            query: search for result of this query
            
        Returns:
            true iff cache answers query
        """
        raise NotImplementedError

    @abstractmethod
    def get_result(self, query):
        """ Get result for query from cache content.
        
        Args:
            query: retrieve result of this query
            
        Returns:
            query result
        """
        raise NotImplementedError

    @abstractmethod
    def update(self):
        """ Update cache content (no effect for static cache). """
        raise NotImplementedError