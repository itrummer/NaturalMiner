'''
Created on Jun 5, 2021

@author: immanueltrummer
'''
from collections import Counter
from cp.pred import is_pred, pred_sql
from dataclasses import dataclass
from typing import FrozenSet
import logging
import time

class AggQuery():
    """ Represents a simple aggregation query. """
    
    def __init__(self, table, eq_preds, cmp_pred, agg_col):
        """ Initializes aggregation query. 
        
        Args:
            table: table the query refers to
            eq_preds: list of equality predicates
            cmp_pred: predicate for comparison
            agg_col: aggregation column
        """
        self.table = table
        self.eq_preds = eq_preds
        self.cmp_pred = cmp_pred
        self.agg_col = agg_col
        
    def pred_cols(self):
        """ Returns columns of query predicates. 
        
        Returns:
            frozen set with predicate columns
        """
        return frozenset([p[0] for p in self.eq_preds if is_pred(p)])
        
    def view(self):
        """ Calculate minimal view containing query result.
        
        Returns:
            query template signature as tuple
        """
        dim_cols = self.pred_cols()
        agg_cols = frozenset([self.agg_col])
        return View(self.table, dim_cols, self.cmp_pred, agg_cols)


@dataclass(frozen=True)
class View():
    """ Represents a materialized view. """
    
    table: str
    dim_cols: FrozenSet[str]
    cmp_pred: str
    agg_cols: FrozenSet[str]
        
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
            return True
        else:
            return False
        
    def merge(self, view):
        """ Generates new view merging this with other view. 
        
        Args:
            view: merge with this view
            
        Returns:
            New view merging dimensions and aggregates
        """
        assert(self.table == view.table)
        assert(self.cmp_pred == view.cmp_pred)
        
        dim_cols = self.dim_cols.union(view.dim_cols)
        agg_cols = self.agg_cols.union(view.agg_cols)
        
        return View(self.table, dim_cols, self.cmp_pred, agg_cols)


class AggCache():
    """ Cache of aggregate values. """

    def __init__(self, connection, src_table, update_every):
        """ Initializes query cache.
        
        Args:
            connection: database connection
            src_table: source of cached data
            update_every: wait between cache updates
        """
        self.connection = connection
        self.update_every = update_every
        self.prefix = 'pcache'
        self.max_cached = 100
        self.v_to_slot = {}
        self.query_log = []
        self._clear_cache()
        self.no_update = 0
        src_view = View(src_table, [], [], [])
        self.miss_penalty = self._estimate_cardinality(src_view)
    
    def can_answer(self, query):
        """ Checks if query can be answered using cached views. 
        
        Args:
            query: check for result of this query
            
        Returns:
            flag indicating if query can be answered from cache
        """
        self.query_log.append(query)
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
        self.query_log.append(query)
        
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
        sql = f'with sums as (' \
            f'select sum(c) as c, sum(s_{query.agg_col}) as s, ' \
            f'sum(cmp_c) as cmp_c, sum(cmp_s_{query.agg_col}) as cmp_s ' \
            f'from {table} where {w_clause}) ' \
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

    def update(self):
        """ Updates cache content for maximum efficiency. """
        self.no_update += 1
        if self.no_update > self.update_every:
            
            views = list(self.v_to_slot.keys())
            candidates = list(self._candidate_views())
            threshold = self.miss_penalty * 5
            v_add = self._select_views(views, candidates, 3, threshold)
            nr_kept = self.max_cached - len(v_add)
            to_keep = self._select_views(list(v_add), views, nr_kept, 0)
            v_del = set(views).difference(to_keep)
            
            logging.debug(f'Query log: {self.query_log}')
            logging.debug(f'Available views: {views}')
            logging.debug(f'View candidates: {candidates}')
            logging.debug(f'Views to add: {v_add}')
            logging.debug(f'Views to remove: {v_del}')
            
            for v in v_del:
                self._drop_results(v)
            for v in v_add:
                self._put_results(v)
            self.query_log.clear()
            self.no_update = 0

    def _candidate_views(self):
        """ Selects candidate views for which to generate results. 
        
        Returns:
            set of candidate views to materialize
        """
        v_counts = Counter()
        for q in self.query_log:
            v = q.view()
            v_counts.update([v])
            
        candidates = set([c[0] for c in v_counts.most_common(10)])
        for _ in range(3):
            expanded = set()
            for v_1 in candidates:
                for v_2 in candidates:
                    v_m = v_1.merge(v_2)
                    expanded.add(v_m)
            candidates.update(expanded)
        
        return candidates

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
        table = view.table
        cols = view.dim_cols
        
        sql = f'explain (format json) select 1 from {table}'
        if cols:
            group_by = ', '.join(cols)
            sql += f' group by {group_by}'

        with self.connection.cursor() as cursor:
            cursor.execute(sql)
            res = cursor.fetchall()
            rows = res[0][0][0]['Plan']['Plan Rows']
            
        return rows

    def _next_slot(self):
        """ Selects next free slot in cache.
        
        Returns:
            lowest slot ID that is available (exception if none)
        """
        return min(set(range(self.max_cached)) - set(self.v_to_slot.values()))

    def _put_results(self, view):
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
            
            s_parts = [f'select count(*) as c,']
            s_parts += [f'sum(case when {cmp_pred} then 1 ' \
                        'else 0 end) as cmp_c']
            
            for agg_col in view.agg_cols:
                s_parts = [f'select sum({agg_col}) as s_{agg_col},']
                s_parts += [f'sum(case when {cmp_pred} then {agg_col} ' \
                            f'else 0 end) as cmp_s_{agg_col}']
                s_parts += list(pred_cols)
            
            q_parts += [', '.join(s_parts)]
            q_parts += [f' from {table}']
            if pred_cols:
                q_parts += [' group by ' + ', '.join(pred_cols)]
            q_parts += [')']
            
            sql = ' '.join(q_parts)
            logging.debug(f'About to fill cache with SQL "{sql}"')
            with self.connection.cursor() as cursor:
                start_s = time.time()
                cursor.execute(sql)
                logging.debug(
                    f'Put time: {time.time() - start_s} seconds ' \
                    f'for view {view}')
            
            self.t_to_slot[view] = slot_id
        
    def _query_cost(self, view, query):
        """ Cost of answering given query from given view. 
        
        Args:
            view: view used for answering query
            query: answer this query using view
            
        Returns:
            estimated cost for answering query
        """
        if view.can_answer(query):
            return self._estimate_cardinality(view)
        else:
            return self.miss_penalty
    
    def _query_log_cost(self, views):
        """ Calculates cost of answering logged queries given views.
        
        Args:
            views: use those views to answer queries
            
        Returns:
            estimated cost for answering queries in log
        """
        cost = 0
        for q in self.query_log:
            default = [self.miss_penalty]
            cost += min([self._query_cost(v, q) for v in views] + default)
        return cost
    
    def _select_views(self, given, candidates, k, threshold):
        """ Select most interesting views to add.
        
        Args:
            given: those views are available
            candidates: select among those views
            k: select so many views greedily
            threshold: add if savings above threshold
            
        Returns:
            near-optimal views to add
        """
        selected = []
        c_left = candidates.copy()
        
        nr_to_add = min(k, len(candidates))
        for _ in range(nr_to_add):
            
            available = given + selected
            c = {v:self._query_log_cost(available + [v]) for v in c_left}
            if c:
                v = min(c, key=c.get)
                old_cost = self._query_log_cost(available)
                new_cost = self._query_log_cost(available + [v])
                savings = old_cost - new_cost
                logging.debug(f'View {v} saves {savings} (T: {threshold})')
                
                if savings >= threshold:
                    selected.append(v)
                    c_left.remove(v)
                else:
                    break
            else:
                break
            
        return set(selected)
    
    def _slot_table(self, slot_id):
        """ Returns name of table storing slot content. 
        
        Args:
            slot_id: slot number
            
        Returns:
            name of table storing cache slot
        """
        return self.prefix + str(slot_id)


class QueryEngine():
    """ Processes queries distinguishing entities from others. """
    
    def __init__(self, connection, table, cmp_pred, update_every):
        """ Initialize query engine for specific connection.
        
        Args:
            connection: connection to database
            table: queries refer to this table
            cmp_pred: use for comparisons
            update_every: pause between updates
        """
        self.connection = connection
        self.table = table
        self.cmp_pred = cmp_pred
        self.connection.autocommit = True
        self.q_cache = AggCache(self.connection, table, update_every)
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
        q_parts = [f'select avg({agg_col}) from {self.table} where TRUE'] 
        q_parts += [pred_sql(col=c, val=v) for c, v in eq_preds]
        q_parts += [pred]
        query = ' AND '.join(q_parts)
        
        with self.connection.cursor() as cursor:
            logging.debug(f'About to execute query {query}')
            cursor.execute(query)
            avg = cursor.fetchone()[0]
            
        return avg
    
    def rel_avg(self, eq_preds, agg_col):
        """ Relative average of focus entity in given data scope. 
        
        Args:
            eq_preds: equality predicates defining scope
            agg_col: consider averages in this column
            
        Returns:
            Ratio of entity to general average
        """
        self.q_cache.update()
        query = AggQuery(self.table, eq_preds, 
                         self.cmp_pred, agg_col)
        
        if self.q_cache.can_answer(query):
            self.cache_hits += 1
            return self.q_cache.get_result(query)
        
        else:
            self.cache_misses += 1
            entity_avg = self.avg(eq_preds, self.cmp_pred, agg_col)
            general_avg = self.avg(eq_preds, 'true', agg_col)
            if entity_avg is None:
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