'''
Created on Aug 4, 2021

@author: immanueltrummer
'''
from collections import defaultdict
from cp.cache.common import AggCache, AggQuery
from cp.fact import Fact
from cp.pred import is_pred, pred_sql
import logging
import time

class ProCache(AggCache):
    """ Implements pro-active caching for RL approach. """
    
    def __init__(self, connection, table, cmp_pred, 
                 nr_facts, nr_preds, all_preds, pred_graph,
                 agg_cols, agg_graph):
        """ Initializes proactive cache.
        
        Args:
            connection: connection to database
            table: name of source table
            cmp_pred: predicate used for comparisons
            nr_facts: number of facts per summary
            nr_preds: number of predicates per fact
            all_preds: list of predicates
            pred_graph: graph of query predicates
            agg_cols: list of aggregation columns
            agg_graph: graph of query aggregates
        """
        self.connection = connection
        self.table = table
        self.cmp_pred = cmp_pred
        self.nr_facts = nr_facts
        self.nr_preds = nr_preds
        self.nr_props = nr_preds + 1
        self.all_preds = all_preds
        self.pred_graph = pred_graph
        self.agg_cols = agg_cols
        self.agg_graph = agg_graph
        self.q_to_r = {}
    
    def can_answer(self, query):
        """ Check if query result is cached.
        
        Args:
            query: look for this query's result
        
        Returns:
            true iff query result is cached
        """
        return query in self.q_to_r
        
    def get_result(self, query):
        """ Get cached result for given query.
        
        Args:
            query: aggregation query for lookup
        
        Returns:
            result for given aggregation query
        """
        return self.q_to_r[query]

    def set_cur_facts(self, cur_facts):
        """ Inform cache about currently selected facts.
        
        Args:
            cur_facts: facts representing RL state
        """
        self.cur_facts = cur_facts

    def update(self):
        """ Proactively cache results close to current facts. """
        q_probs = [(q, p) for q, p in self._query_probs(1).items()]
        for fact in self.cur_facts:
            query = self._props_query(fact.props)
            if not self.can_answer(query):
                
                p_q_probs = list(filter(
                    lambda q:not self.can_answer(q), q_probs))
                logging.debug(f'Query probs: {p_q_probs}')
                r_aggs = self._rank_aggs(p_q_probs)
                logging.debug(f'Ranked aggregates: {r_aggs}')
                r_preds = self._rank_preds(p_q_probs, query)
                logging.debug(f'Ranked predicates: {r_preds}')
                
                aggs = {query.agg_col}
                preds = {frozenset(query.eq_preds)}
                n_aggs = min(len(r_aggs), 3)
                n_preds = min(len(r_preds), 5)
                aggs += [a[0] for a in r_aggs[0:n_aggs]]
                preds += [p[0] for p in r_preds[0:n_preds]]
                logging.debug(f'Selected aggs: {aggs}')
                logging.debug(f'Selected preds: {preds}')
                self._cache(aggs, preds)
    
    def _cache(self, aggs, preds):
        """ Cache results for given aggregates and predicates. 
        
        Args:
            aggs: aggregates to cache
            preds: predicates to cache
        """
        q_parts = [self._sql_select(aggs, preds)]
        q_parts += [f' from {self.table}']
        q_parts += [self._sql_where(preds)]
        q_parts += [self._sql_group(preds)]
        sql = ' '.join(q_parts)
        logging.debug(f'About to fill cache with SQL "{sql}"')
        
        with self.connection.cursor() as cursor:
            start_s = time.time()
            cursor.execute(sql)
            total_s = time.time() - start_s
            logging.debug(f'Time: {total_s} s for query {sql}')
            rows = cursor.fetchall()
            self._extract_results(aggs, preds, rows)
    
    def _extract_results(self, aggs, preds, rows):
        """ Extracts new cache entries from query result.
        
        Args:
            aggs: aggregation columns
            preds: predicate groups used for query
            rows: result rows of caching query
        """
        if preds:
            dims = {p[0] for p in next(iter(preds))}
        else:
            dims = {}

        for r in rows:
            cmp_c = r['cmp_c']
            if cmp_c > 0:
                c = r['c']
                for agg in aggs:
                    s = r[f's_{agg}']
                    if s > 0:
                        cmp_s = r[f'cmp_s_{agg}']
                        eq_preds = [(d, r[d]) for d in dims]
                        q = AggQuery(self.table, frozenset(eq_preds), 
                                     self.cmp_pred, agg)
                        rel_avg = (cmp_s/cmp_c)/(s/c)
                        self.q_to_r[q] = rel_avg
    
    def _query_probs(self, max_steps):
        """ Calculates probability that facts become relevant in few steps.
        
        Args:
            max_steps: consider at most that many steps
        
        Returns:
            Dictionary mapping queries to probabilities
        """
        fact_prob = 1.0/self.nr_facts
        for fact in self.cur_facts:
            p_by_step = []
            p_by_step += [{tuple(fact.props):1}]
            for _ in range(max_steps):
                
                p_last = p_by_step[-1]
                p_next = defaultdict(lambda:0)
                p_by_step.append(p_next)
                
                for s_props, s_prob in p_last.items():
                    for prop in range(self.nr_props):
                        # print(f'nr_props: {self.nr_props}')
                        # print(f's_props: {s_props}')
                        val = s_props[prop]
                        if prop == self.nr_preds:
                            n_vals = self.agg_graph.get_reachable(val, 1)
                        else:
                            n_vals = self.pred_graph.get_reachable(val, 1)
                        
                        for n_val in n_vals:
                            n_props = [p for p in s_props]
                            n_props[prop] = n_val
                            n_props = tuple(n_props)
                            val_prob = 1.0/len(n_vals)
                            t_prob = s_prob * fact_prob * val_prob
                            p_next[n_props] += t_prob
        
        agg_probs = defaultdict(lambda:0)
        for step_probs in p_by_step:
            for props, prob in step_probs.items():
                query = self._props_query(props)
                agg_probs[query] += prob

        return agg_probs

    def _props_query(self, props):
        """ Generate aggregation query for property tuple. 
        
        Args:
            props: tuple of properties describing fact
            
        Returns:
            aggregation query associated with fact properties
        """
        fact = Fact.from_props(list(props))
        eq_preds = [self.all_preds[p] for p in fact.get_preds()]
        eq_preds = list(filter(lambda p:is_pred(p), eq_preds))
        agg_col = self.agg_cols[fact.get_agg()]
        return AggQuery(self.table, frozenset(eq_preds), self.cmp_pred, agg_col)
    
    def _rank_aggs(self, q_probs):
        """ Rank aggregates by their probability.
        
        Args:
            q_probs: pairs of queries and probabilities
            
        Returns:
            list of aggregates ordered by probability (descending)
        """
        p_agg = defaultdict(lambda:0)
        for q, p in q_probs:
            agg = q.agg_col
            p_agg[agg] += p
            
        return sorted(p_agg.items(), key=lambda i:i[1], reverse=True)
    
    def _rank_preds(self, q_probs, query):
        """ Rank query-compatible predicates based on probabilities.
        
        Args:
            q_probs: pairs of queries and probabilities
            query: rank predicates with same signature as in query
        
        Returns:
            list of compatible predicate groups by probability (descending)
        """
        q_dims = query.pred_cols()
        p_pred = defaultdict(lambda:0)
        for q, p in q_probs:
            p_dims = q.pred_cols()
            if p_dims == q_dims:
                preds = frozenset(q.eq_preds)
                p_pred[preds] += p
        
        return sorted(p_pred.items(), key=lambda i:i[1], reverse=True)
    
    def _sql_select(self, aggs, preds):
        """ Generates select clause of cache query.
        
        Args:
            aggs: list of aggregate columns
            preds: list of predicate groups
        
        Returns:
            SQL select clause for query
        """
        s_parts = [f'select count(*) as c']
        s_parts += [f'sum(case when {self.cmp_pred} then 1 ' \
                    'else 0 end) as cmp_c']
        
        if preds:
            pred_cols = {p[0] for p in next(iter(preds))}
            s_parts += list(pred_cols)
        
        for agg_col in aggs:
            s_parts += [f'sum({agg_col}) as s_{agg_col}']
            s_parts += [f'sum(case when {self.cmp_pred} then {agg_col} ' \
                        f'else 0 end) as cmp_s_{agg_col}']
        
        return ', '.join(s_parts)
    
    def _sql_group(self, preds):
        """  Generates group-by clause of cache query.
        
        Args:
            preds: list of predicate sets
            
        Returns:
            SQL group-by clause for query
        """
        if preds:
            dim_cols = {p[0] for p in next(iter(preds))}
            if dim_cols:
                return 'group by ' + ', '.join(dim_cols)
        return ''
    
    def _sql_where(self, preds):
        """ Generates where clause of cache query.
        
        Args:
            preds: list of predicate sets
            
        Returns:
            SQL where clause for query
        """
        w_parts = []
        for p_group in preds:
            if p_group:
                c_pred = ' or '.join([pred_sql(p, v) for p, v in p_group])
                w_parts += ['(' + c_pred + ')']

        if w_parts:
            return ' where ' + ' and '.join(w_parts)
        else:
            return ''