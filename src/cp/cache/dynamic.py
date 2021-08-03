'''
Created on Jul 31, 2021

@author: immanueltrummer
'''
from cp.cache.common import AggCache
from cp.pred import pred_sql
from dataclasses import dataclass
from typing import FrozenSet, Tuple
import logging
import time

@dataclass(frozen=True)
class View():
    """ Represents a materialized view. """
    
    table: str
    dim_cols: FrozenSet[str]
    cmp_pred: str
    agg_cols: FrozenSet[str]
    scope: FrozenSet[Tuple[str, FrozenSet[str]]]

    @staticmethod    
    def from_query(query, im_scope):
        """ Calculate minimal view containing query result.
        
        Args:
            query: generate view subsuming this query
            im_scope: immutable data scope
        
        Returns:
            query template signature as tuple
        """
        dim_cols = query.pred_cols()
        agg_cols = frozenset([query.agg_col])
        return View(query.table, dim_cols, query.cmp_pred, agg_cols, im_scope)
        
    def can_answer(self, query):
        """ Determines if view contains answer to query. 
        
        Args:
            query: test if view can answer this query
            
        Returns:
            true iff the view can answer the query
        """
        
        if self.table == query.table and \
            self.cmp_pred == query.cmp_pred and \
            query.agg_col in self.agg_cols and \
            query.pred_cols().issubset(self.dim_cols):
            
            for q_d, q_v in query.eq_preds:
                for s_d, s_vals in self.scope:
                    if q_d == s_d and not (q_v in s_vals):
                        return False
            
            return True
        else:
            return False
            
    def merge(self, view, im_scope):
        """ Generates new view merging this with other view. 
        
        Args:
            view: merge with this view
            im_scope: immutable scope
            
        Returns:
            New view merging dimensions and aggregates
        """
        assert(self.table == view.table)
        assert(self.cmp_pred == view.cmp_pred)
        
        dim_cols = self.dim_cols.union(view.dim_cols)
        agg_cols = self.agg_cols.union(view.agg_cols)
        
        return View(self.table, dim_cols, self.cmp_pred, agg_cols, im_scope)


class DynamicCache(AggCache):
    """ Cache of aggregate values, updating cache content dynamically. """

    def __init__(self, connection):
        """ Initializes query cache.
        
        Args:
            connection: database connection
        """
        self.connection = connection
        self.prefix = 'pcache'
        self.max_cached = 100
        self.v_to_slot = {}
        self._clear_cache()
    
    def can_answer(self, query):
        """ Checks if query can be answered using cached views. 
        
        Args:
            query: check for result of this query
            
        Returns:
            flag indicating if query can be answered from cache
        """
        views = self.v_to_slot.keys()
        if [v for v in views if v.can_answer(query)]:
            return True
        else:
            return False

    def get_result(self, query):
        """ Generate query result from a view. 
        
        Args:
            query: retrieve result for this query
            
        Returns:
            query result (a relative average)
        """
        q_views = []
        for v in self.v_to_slot.keys():
            if v.can_answer(query):
                q_views.append(v)
                
        v_cards = {v:self._estimate_cardinality(v) for v in q_views}
        view = min(v_cards, key=v_cards.get)
            
        slot_id = self.v_to_slot[view]
        table = self._slot_table(slot_id)
        
        p_parts = [pred_sql(c,v) for c, v in query.eq_preds]
        w_clause = ' and '.join(p_parts)
        w_clause = 'where ' + w_clause if w_clause else w_clause
        sql = f'with sums as (' \
            f'select sum(c) as c, sum(s_{query.agg_col}) as s, ' \
            f'sum(cmp_c) as cmp_c, sum(cmp_s_{query.agg_col}) as cmp_s ' \
            f'from {table} {w_clause}) ' \
            f'select case when cmp_c = 0 or s = 0 then NULL '\
            f'else (cmp_s/cmp_c)/(s/c) end as rel_avg from sums'
        
        with self.connection.cursor() as cursor:
            start_s = time.time()
            cursor.execute(sql)
            logging.debug(f'Get time: {time.time() - start_s} seconds')
            if cursor.rowcount == 0:
                return None
            else:
                return cursor.fetchone()[0]

    def put_results(self, view):
        """ Generates and register materialized view.
        
        Args:
            view: generate this view
        """
        if view not in self.v_to_slot:
            
            slot_id = self._next_slot()
            table = view.table
            cmp_pred = view.cmp_pred
            pred_cols = view.dim_cols
            
            cache_tbl = self._slot_table(slot_id)
            q_parts = [f'create unlogged table {cache_tbl} as (']
            
            s_parts = [f'select count(*) as c']
            s_parts += [f'sum(case when {cmp_pred} then 1 ' \
                        'else 0 end) as cmp_c']
            s_parts += list(pred_cols)
            
            for agg_col in view.agg_cols:
                s_parts += [f'sum({agg_col}) as s_{agg_col}']
                s_parts += [f'sum(case when {cmp_pred} then {agg_col} ' \
                            f'else 0 end) as cmp_s_{agg_col}']
            
            q_parts += [', '.join(s_parts)]
            q_parts += [f' from {table}']
            q_parts += [self._view_where_sql(view)]
            q_parts += [self._view_group_sql(view)]
            q_parts += [')']

            sql = ' '.join(q_parts)
            logging.debug(f'About to fill cache with SQL "{sql}"')
            with self.connection.cursor() as cursor:
                start_s = time.time()
                cursor.execute(sql)
                logging.debug(
                    f'Put time: {time.time() - start_s} seconds ' \
                    f'for view {view}')
            
            self.v_to_slot[view] = slot_id

    def update(self):
        pass

    def _clear_cache(self):
        """ Clears all cached relations. """
        for i in range(self.max_cached):
            with self.connection.cursor() as cursor:
                cache_tbl = self._slot_table(i)
                sql = f'drop table if exists {cache_tbl}'
                cursor.execute(sql)

    def _drop_results(self, view):
        """ Drop given view. 
        
        Args:
            template: drop results for this view
        """
        slot_id = self.v_to_slot[view]
        slot_tbl = self._slot_table(slot_id)
        with self.connection.cursor() as cursor:
            cursor.execute(f'drop table if exists {slot_tbl}')
        del self.v_to_slot[view]
        
    def _estimate_cardinality(self, view):
        """ Estimate cardinality for given view. 
        
        Args:
            view: analyze cardinality of this view
            
        Returns:
            estimated cardinality for view
        """
        q_parts = [f'explain (format json) select 1 from {view.table}']
        q_parts += [self._view_where_sql(view)]
        q_parts += [self._view_group_sql(view)]
        sql = ' '.join(q_parts)
        
        return self._row_estimate(sql)

    def _next_slot(self):
        """ Selects next free slot in cache.
        
        Returns:
            lowest slot ID that is available (exception if none)
        """
        return min(set(range(self.max_cached)) - set(self.v_to_slot.values()))

    def _row_estimate(self, explain_sql):
        """ Extracts result row estimate for explain query.
        
        Args:
            explain_sql: SQL explain query (as string)
            
        Returns:
            estimated number of query result rows
        """
        with self.connection.cursor() as cursor:
            cursor.execute(explain_sql)
            res = cursor.fetchall()
            rows = res[0][0][0]['Plan']['Plan Rows']
            
        return rows

    def _slot_table(self, slot_id):
        """ Returns name of table storing slot content. 
        
        Args:
            slot_id: slot number
            
        Returns:
            name of table storing cache slot
        """
        return self.prefix + str(slot_id)
    
    def _view_group_sql(self, view):
        """  Generates group-by clause of query generating view.
        
        Args:
            view: a view
            
        Returns:
            SQL group-by clause for view
        """
        dim_cols = view.dim_cols
        if dim_cols:
            return 'group by ' + ', '.join(dim_cols)
        else:
            return ''
    
    def _view_where_sql(self, view):
        """ Generates where clause of query generating view.
        
        Args:
            view: a view
            
        Returns:
            SQL where clause for view
        """
        w_parts = []
        for s_d, s_vals in view.scope:
            if s_d in view.dim_cols:
                c_parts = [pred_sql(s_d, v) for v in s_vals]
                if c_parts:
                    w_parts += ['(' + ' or '.join(c_parts) + ')']

        if w_parts:
            return ' where ' + ' and '.join(w_parts)
        else:
            return ''