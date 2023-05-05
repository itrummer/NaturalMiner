'''
Created on Jun 5, 2021

@author: immanueltrummer
'''
from nminer.sql.pred import is_pred, pred_sql
from dataclasses import dataclass
from typing import FrozenSet, Tuple, Set

import logging
import time

@dataclass(frozen=True)
class AggQuery():
    """ Represents a simple aggregation query. """
    
    table: str
    eq_preds: FrozenSet[Tuple[str, str]]
    cmp_pred: str
    agg_col: str
    
    @staticmethod
    def from_fact(table, all_preds, cmp_pred, agg_cols, fact):
        """ Generate query associated with given fact.
        
        Args:
            table: fact refers to data in this table
            all_preds: list of all predicates
            cmp_pred: predicate for comparisons
            agg_cols: aggregation columns
            fact: construct query for this fact
        
        Returns:
            aggregation query
        """
        eq_preds = [all_preds[p] for p in fact.get_preds()]
        eq_preds = list(filter(lambda p:is_pred(p), eq_preds))
        eq_preds = frozenset(eq_preds)
        agg_idx = fact.get_agg()
        agg_col = agg_cols[agg_idx]
        return AggQuery(table, eq_preds, cmp_pred, agg_col)

    def pred_cols(self):
        """ Returns columns of query predicates. 
        
        Returns:
            frozen set with predicate columns
        """
        return frozenset([p[0] for p in self.eq_preds if is_pred(p)])


@dataclass
class GroupQuery():
    """ Represents a group-by aggregation query. """
    
    table: str
    dims: Set[str]
    cmp_pred: str
    aggs: Set[str]
    preds: Set[Tuple[str, str]]
    
    def __init__(self, table, dims, cmp_pred):
        """ Initializes group queries.
        
        Args:
            table: queries refer to this table
            dims: group by these columns
            cmp_pred: predicate for comparisons
        """
        self.table = table
        self.dims = dims
        self.cmp_pred = cmp_pred
        self.aggs = set()
        self.preds = set()
    
    @staticmethod
    def from_query(query):
        """ Initializes group-by query from simple aggregation query. 
        
        Args:
            query: simple aggregation query
        """
        g_q = GroupQuery(query.table, query.pred_cols(), query.cmp_pred)
        g_q.integrate(query)
        return g_q
    
    def can_merge(self, query):
        """ Check if simple query can be merged into group-by query.
        
        Args:
            query: simple aggregation query
        
        Returns:
            true iff the given query can be merged
        """
        return True if query.pred_cols == self.dims else False
    
    def contains(self, query):
        """ Check if group-by query generates result to simple query.
        
        Args:
            query: a simple aggregation query
        
        Returns:
            true iff the simple query is contained
        """
        if query.agg_col in self.aggs and query.eq_preds in self.preds and \
            query.table == self.table and query.cmp_pred == self.cmp_pred:
            return True
        else:
            return False
    
    def integrate(self, query):
        """ Expand group-by query to subsume given query.
        
        Args:
            query: integrate this aggregation query
        """
        assert(query.table == self.table)
        assert(query.pred_cols() == frozenset(self.dims))
        assert(query.cmp_pred == self.cmp_pred)
        
        self.aggs.add(query.agg_col)
        self.preds.add(query.eq_preds)
    
    def result_size(self):
        """ Calculates maximal result cardinality.
        
        Returns:
            Maximal result cardinality
        """
        return len(self.aggs) * len(self.preds)
    
    def sql(self):
        """ Generates equivalent SQL query.
        
        Returns:
            SQL representation of query
        """
        q_parts = [self._sql_select()]
        q_parts += [f' from {self.table}']
        q_parts += [self._sql_where()]
        q_parts += [self._sql_group()]
        return ' '.join(q_parts)

    def _sql_select(self):
        """ Generates select clause of cache query.
        
        Returns:
            SQL select clause for query
        """
        s_parts = []
        if self.dims:
            s_parts += list(self.dims)
        
        for agg_col in self.aggs:
            s_parts += [f'count({agg_col}) as c_{agg_col}']
            s_parts += [f'sum(case when {self.cmp_pred} ' \
                        f'and {agg_col} is not null then 1 ' \
                        f'else 0 end) as cmp_c_{agg_col}']
            s_parts += [f'sum({agg_col}) as s_{agg_col}']
            s_parts += [f'sum(case when {self.cmp_pred} then {agg_col} ' \
                        f'else 0 end) as cmp_s_{agg_col}']
        
        return 'select ' + ', '.join(s_parts)
    
    def _sql_group(self):
        """  Generates group-by clause of cache query.
            
        Returns:
            SQL group-by clause for query
        """
        if self.dims:
            return 'group by ' + ', '.join(self.dims)
        else:
            return ''
    
    def _sql_where(self):
        """ Generates where clause of cache query.
            
        Returns:
            SQL where clause for query
        """
        w_parts = []
        for p_group in self.preds:
            if p_group:
                c_pred = ' and '.join([pred_sql(p, v) for p, v in p_group])
                w_parts += ['(' + c_pred + ')']

        if w_parts:
            return ' where ' + ' or '.join(w_parts)
        else:
            return ''
 
    
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
        self.q_cache = cache
        self.nr_queries = 0
        
    def avg(self, eq_preds, pred, agg_col):
        """ Calculate average over aggregation column in scope. 
        
        Args:
            eq_preds: equality predicates as column-value pairs
            pred: SQL string representing predicate
            agg_col: calculate average for this column
            
        Returns:
            Average over aggregation column for satisfying rows, row count
        """
        q_parts = [f'select avg({agg_col}) as avg, ' \
                   f'count(*) as count from {self.table} where TRUE']
        q_parts += [pred_sql(col=c, val=v) for c, v in eq_preds]
        q_parts += [pred]
        query = ' AND '.join(q_parts)
        
        with self.connection.cursor() as cursor:
            logging.debug(f'About to execute query {query}')
            cursor.execute(query)
            row = cursor.fetchone()
            avg = row['avg']
            count = row['count']

        return avg, count
    
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
            logging.debug(f'Cache hit: {query}')
            return self.q_cache.get_result(query)        
        else:
            logging.debug(f'Cache miss: {query}')
            self.nr_queries += 1
            start_s = time.time()
            e_avg, e_cnt = self.avg(eq_preds, self.cmp_pred, agg_col)
            g_avg, _ = self.avg(eq_preds, 'true', agg_col)
            total_s = time.time() - start_s
            logging.debug(f'Processing {query} took {total_s} seconds')
            
            if e_avg is None or g_avg is None:
                return None, None
            else:
                f_gen_avg = max(0.0001, float(g_avg))
                rel_avg = float(e_avg) / f_gen_avg
                return rel_avg, e_cnt
            
    def statistics(self):
        """ Generates performance statistics. 
        
        Returns:
            Dictionary with statistics
        """
        return {'nr_queries':self.nr_queries}