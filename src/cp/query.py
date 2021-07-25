'''
Created on Jun 5, 2021

@author: immanueltrummer
'''
from collections import Counter
from cp.pred import is_pred, pred_sql
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
        
    def template(self):
        """ Calculate query template signature for cache lookups.
        
        Returns:
            query template signature as tuple
        """
        cols = self.pred_cols()
        return self.table, cols, self.cmp_pred, self.agg_col


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
        self.t_to_slot = {}
        self.query_log = []
        self._clear_cache()
        self.no_update = 0        
        self.miss_penalty = self._estimate_cardinality(
            (src_table, [], [], []))
    
    def can_answer(self, query):
        """ Checks if query can be answered using cached views. 
        
        Args:
            query: check for result of this query
            
        Returns:
            flag indicating if query can be answered from cache
        """
        self.query_log.append(query)
        t = query.template()
        views = self.t_to_slot.keys()
        if [v for v in views if self._specializes(t, v)]:
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
        t = query.template()
        for v in self.t_to_slot.keys():
            if self._specializes(t, v):
                q_views.append(v)
                
        v_cards = {v:self._estimate_cardinality(v) for v in q_views}
        view = min(v_cards, key=v_cards.get)
            
        slot_id = self.t_to_slot[view]
        table = self._slot_table(slot_id)
        
        p_parts = [pred_sql(c,v) for c, v in query.eq_preds]
        w_clause = ' and '.join(p_parts)
        sql = f'with sums as (' \
            f'select sum(c) as c, sum(s) as s, ' \
            f'sum(cmp_c) as cmp_c, sum(cmp_s) as cmp_s ' \
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
            views = list(self.t_to_slot.keys())
            candidates = list(self._candidate_views())
            v_add = self._select_views(views, candidates, 3)
            nr_kept = self.max_cached - len(v_add)
            to_keep = self._select_views(candidates, views, nr_kept)
            v_del = set(views).difference(to_keep)
            
            for v in v_del:
                self._drop_results(v)
            for v in v_add:
                self._put_results(v)
            self.query_log.clear()
            self.no_update = 0

    def _candidate_views(self):
        """ Selects candidate templates for which to generate results. 
        
        Returns:
            set of query templates representing candidates
        """
        t_counts = Counter()
        for q in self.query_log:            
            t = q.template()
            t_counts.update([t])
            
        candidates = set([c[0] for c in t_counts.most_common(10)])
        for _ in range(3):
            expanded = set()
            for t_1 in candidates:
                for t_2 in candidates:
                    t_m = self._merge_templates(t_1, t_2)
                    if t_m:
                        expanded.add(t_m)
            candidates.update(expanded)
        
        return candidates

    def _clear_cache(self):
        """ Clears all cached relations. """
        for i in range(self.max_cached):
            with self.connection.cursor() as cursor:
                cache_tbl = self._slot_table(i)
                sql = f'drop table if exists {cache_tbl}'
                cursor.execute(sql)

    def _drop_results(self, template):
        """ Drop view for given template. 
        
        Args:
            template: drop results for this view
        """
        slot_id = self.t_to_slot[template]
        slot_tbl = self._slot_table(slot_id)
        with self.connection.cursor() as cursor:
            cursor.execute(f'drop table if exists {slot_tbl}')
        del self.t_to_slot[template]

    def _estimate_cardinality(self, template):
        """ Estimate cardinality for view storing template results. 
        
        Args:
            template: analyze view associated with this template
            
        Returns:
            estimated cardinality for view on template
        """
        table, cols, _, _ = template
        sql = f'explain (format json) select 1 from {table}'
        if cols:
            group_by = ', '.join(cols)
            sql += f' group by {group_by}'

        with self.connection.cursor() as cursor:
            cursor.execute(sql)
            res = cursor.fetchall()
            rows = res[0][0][0]['Plan']['Plan Rows']
            
        return rows

    def _merge_templates(self, t_1, t_2):
        """ Merge templates into a new template.
        
        Args:
            t_1: first template to merge
            t_2: second template to merge
            
        Returns:
            template combining columns if compatible, otherwise None
        """
        table_1, col_1, pred_1, agg_1 = t_1
        table_2, col_2, pred_2, agg_2 = t_2
        if table_1 == table_2 and pred_1 == pred_2 and agg_1 == agg_2:
            merged_cols = col_1.union(col_2)
            return table_1, merged_cols, pred_1, agg_1
        else:
            return None

    def _next_slot(self):
        """ Selects next free slot in cache.
        
        Returns:
            lowest slot ID that is available (exception if none)
        """
        return min(set(range(self.max_cached)) - set(self.t_to_slot.values()))

    def _put_results(self, template):
        """ Generates and caches results for query template.
        
        Args:
            template: query template for which to store results
        """
        if template not in self.t_to_slot:
            
            slot_id = self._next_slot()
            table, pred_cols, cmp_pred, agg_col = template
            cache_tbl = self._slot_table(slot_id)
            q_parts = [f'create unlogged table {cache_tbl} as (']
            
            s_parts = [f'select sum({agg_col}) as s, count(*) as c']
            s_parts += [f'sum(case when {cmp_pred} then {agg_col} ' \
                        'else 0 end) as cmp_s']
            s_parts += [f'sum(case when {cmp_pred} then 1 ' \
                        'else 0 end) as cmp_c']
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
                logging.debug(f'Put time: {time.time() - start_s} seconds')
            
            self.t_to_slot[template] = slot_id
        
    def _query_cost(self, view, query):
        """ Cost of answering given query from given view. 
        
        Args:
            view: view used for answering query
            query: answer this query using view
            
        Returns:
            estimated cost for answering query, 'inf' if impossible
        """
        if self._specializes(query.template(), view):
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
            cost += min([self._query_cost(v, q) for v in views])
        return cost
    
    def _select_views(self, given, candidates, k):
        """ Select most interesting views to add.
        
        Args:
            given: those views are available
            candidates: select among those views
            k: select so many views greedily
            
        Returns:
            near-optimal views to add
        """
        selected = []
        nr_to_add = min(k, len(candidates))
        for _ in range(nr_to_add):
            available = given + selected
            c = {v:self._query_log_cost(available + [v]) for v in candidates}
            v = min(c, key=c.get)
            selected.append(v)
        return set(selected)
    
    def _slot_table(self, slot_id):
        """ Returns name of table storing slot content. 
        
        Args:
            slot_id: slot number
            
        Returns:
            name of table storing cache slot
        """
        return self.prefix + str(slot_id)
    
    def _specializes(self, t_1, t_2):
        """ Determines if first template specializes second. 
        
        Args:
            t_1: check if this template specializes the other
            t_2: check if this template generalizes the other
            
        Returns:
            true iff the first template specializes the second
        """
        table_1, cols_1, pred_1, agg_1 = t_1
        table_2, cols_2, pred_2, agg_2 = t_2
        if table_1 == table_2 and pred_1 == pred_2 and \
            agg_1 == agg_2 and cols_1.issubset(cols_2):
            return True
        else:
            return False


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