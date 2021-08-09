'''
Created on Jul 31, 2021

@author: immanueltrummer
'''
from abc import ABC, abstractmethod

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
            pair of relative average and, optionally, row count
        """
        raise NotImplementedError

    @abstractmethod
    def update(self):
        """ Update cache content (no effect for static cache). """
        raise NotImplementedError