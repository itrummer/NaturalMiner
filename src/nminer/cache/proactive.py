'''
Created on Aug 4, 2021

@author: immanueltrummer
'''
from collections import defaultdict
from nminer.sql.cost import estimates
from nminer.sql.query import AggQuery, GroupQuery
from nminer.cache.dynamic import DynamicCache
from nminer.text.fact import Fact
import logging

class ProCache(DynamicCache):
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
        super().__init__(connection, table, cmp_pred)
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
        self.nr_hits = 0
        self.nr_miss = 0

    def set_cur_facts(self, cur_facts):
        """ Inform cache about currently selected facts.
        
        Args:
            cur_facts: facts representing RL state
        """
        self.cur_facts = cur_facts

    def statistics(self):
        """ Returns number of cache hits and misses. """
        return {'cache_hits':self.nr_hits, 'cache_misses':self.nr_miss}

    def update(self):
        """ Proactively cache results close to current facts. """
        q_probs = [(q, p) for q, p in self._query_probs(1).items()]
        for fact in self.cur_facts:
            query = self._props_query(fact.props)
            if not self.can_answer(query):
                self.nr_miss += 1
                logging.debug(f'Query to expand: {query}')
                u_q_probs = [q_p for q_p in q_probs 
                             if not self.can_answer(q_p[0])]
                logging.debug(f'Uncached query probs: {u_q_probs}')
                r_aggs = self._rank_aggs(u_q_probs)
                logging.debug(f'Ranked aggregates: {r_aggs}')
                r_preds = self._rank_preds(u_q_probs, query)
                logging.debug(f'Ranked predicates: {r_preds}')
                exp_g_q = self._expand(query, u_q_probs, 1.5)
                logging.debug(f'Expanded query: {exp_g_q}')
                self.cache(exp_g_q)
            else:
                self.nr_hits += 1
    
    def _coverage(self, g_query, q_probs):
        """ Probability sum of queries covered by group-by query.
        
        Args:
            g_query: a group-by query covering multiple simple queries
            q_probs: list of pairs (queries with associated probabilities)
        
        Returns:
            sum of probability over all covered queries
        """
        agg_prob = 0
        for q, p in q_probs:
            if g_query.contains(q):
                agg_prob += p
        
        return agg_prob
    
    def _expand(self, query, q_probs, max_cost):
        """ Expands given query under cost constraint.
        
        Args:
            query: expand this query
            q_probs: other queries to cover with probabilities
            max_const: maximal relative execution cost
        
        Returns:
            group-by query covering multiple simple queries
        """
        r_aggs = self._rank_aggs(q_probs)
        logging.debug(f'Ranked aggregates: {r_aggs}')
        r_preds = self._rank_preds(q_probs, query)
        logging.debug(f'Ranked predicates: {r_preds}')
        
        g_query = GroupQuery.from_query(query)
        g_sql = g_query.sql()
        _, base_cost = estimates(self.connection, g_sql)
        
        max_cover = 0
        max_exp = g_query
        nr_aggs = len(r_aggs)
        for aggs_ctr in range(nr_aggs):
            aggs = r_aggs[0:aggs_ctr+1]
            e_query = GroupQuery.from_query(query)
            e_query.aggs = aggs
            
            for p in r_preds:
                e_query.preds.add(p)
                e_sql = e_query.sql()
                _, e_cost = estimates(self.connection, e_sql)
                if e_cost <= max_cost * base_cost:
                    cover = self._coverage(e_query, q_probs)
                    if cover > max_cover:
                        max_cover = cover
                        max_exp = e_query
        
        logging.debug(f'Max cover {max_cover} via {max_exp}')
        return max_exp

    def _neighbors(self, s_props):
        """ Generates neighbors in search graph for given properties.
        
        Args:
            s_props: start node in search graph (properties describing fact)
        
        Returns:
            set of neighbors in search graph
        """
        neighbors = []
        for prop in range(self.nr_props):
            
            val = s_props[prop]
            if prop == self.nr_preds:
                n_vals = self.agg_graph.get_reachable(val, 1)
            else:
                n_vals = self.pred_graph.get_reachable(val, 1)
            
            for n_val in n_vals:
                n_props = [p for p in s_props]
                n_props[prop] = n_val
                n_props = tuple(n_props)
                neighbors.append(n_props)
        
        return neighbors
    
    def _props_query(self, props):
        """ Generate aggregation query for property tuple. 
        
        Args:
            props: tuple of properties describing fact
            
        Returns:
            aggregation query associated with fact properties
        """
        fact = Fact.from_props(list(props))
        return AggQuery.from_fact(
            self.table, self.all_preds, 
            self.cmp_pred, self.agg_cols, fact)

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
                    neighbors = self._neighbors(s_props)
                    logging.debug(f'Neighbors of {s_props}: {neighbors}')
                    
                    # Probability of start state, fact selection, and end state
                    n_prob = 1.0/len(neighbors)
                    t_prob = s_prob * fact_prob * n_prob
                    for n_probs in neighbors:
                        p_next[n_probs] += t_prob
        
        agg_probs = defaultdict(lambda:0)
        for step_probs in p_by_step:
            for props, prob in step_probs.items():
                query = self._props_query(props)
                agg_probs[query] += prob

        return agg_probs

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
            
        agg_p_sorted = sorted(p_agg.items(), key=lambda i:i[1], reverse=True)
        return [a for a, _ in agg_p_sorted]
    
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
        
        pred_p_sorted = sorted(p_pred.items(), key=lambda i:i[1], reverse=True)
        return [p for p, _ in pred_p_sorted]