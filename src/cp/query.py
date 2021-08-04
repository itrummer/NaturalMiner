'''
Created on Jun 5, 2021

@author: immanueltrummer
'''
from cp.cache.common import AggQuery
from cp.pred import pred_sql
import logging
import time

class QueryEngine():
    """ Processes queries distinguishing entities from others. """
    
    def __init__(self, connection, table, cmp_pred, cache):
        """ Initialize query engine for specific connection.
        
        Args:
            connection: connection to database
            table: queries refer to this table
            cmp_pred: use for comparisons
            cache: cache for query results
        """
        self.connection = connection
        self.table = table
        self.cmp_pred = cmp_pred
        self.connection.autocommit = True
        self.q_cache = cache
        self.cache_hits = 0
        self.cache_misses = 0
        
    def avg(self, eq_preds, pred, agg_col):
        """ Calculate average over aggregation column in scope. 
        
        Args:
            eq_preds: equality predicates as column-value pairs
            pred: SQL string representing predicate
            agg_col: calculate average for this column
            
        Returns:
            Average over aggregation column for satisfying rows
        """
        q_parts = [f'select avg({agg_col}) as avg from {self.table} where TRUE'] 
        q_parts += [pred_sql(col=c, val=v) for c, v in eq_preds]
        q_parts += [pred]
        query = ' AND '.join(q_parts)
        
        with self.connection.cursor() as cursor:
            logging.debug(f'About to execute query {query}')
            cursor.execute(query)
            avg = cursor.fetchone()['avg']
            
        return avg
    
    def rel_avg(self, eq_preds, agg_col):
        """ Relative average of focus entity in given data scope. 
        
        Args:
            eq_preds: equality predicates defining scope
            agg_col: consider averages in this column
            
        Returns:
            Ratio of entity to general average
        """
        query = AggQuery(self.table, frozenset(eq_preds), 
                         self.cmp_pred, agg_col)
        
        if self.q_cache.can_answer(query):
            self.cache_hits += 1
            logging.debug(f'Cache hit: {query}')
            return self.q_cache.get_result(query)
        
        else:
            self.cache_misses += 1
            logging.debug(f'Cache miss: {query}')
            
            start_s = time.time()
            entity_avg = self.avg(eq_preds, self.cmp_pred, agg_col)
            general_avg = self.avg(eq_preds, 'true', agg_col)
            total_s = time.time() - start_s
            logging.debug(f'Processing {query} took {total_s} seconds')
            
            if entity_avg is None or general_avg is None:
                return None
            else:
                f_gen_avg = max(0.0001, float(general_avg))
                return float(entity_avg) / f_gen_avg
            
    def statistics(self):
        """ Generates performance statistics. 
        
        Returns:
            Dictionary with statistics
        """
        return {'cache_hits':self.cache_hits, 'cache_misses':self.cache_misses}