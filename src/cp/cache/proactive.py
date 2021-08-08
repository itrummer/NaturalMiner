'''
Created on Aug 4, 2021

@author: immanueltrummer
'''
from collections import defaultdict
from cp.sql.query import AggQuery
from cp.cache.dynamic import DynamicCache
from cp.text.fact import Fact
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
                aggs.update([a[0] for a in r_aggs[0:n_aggs]])
                preds.update([p[0] for p in r_preds[0:n_preds]])
                logging.debug(f'Selected aggs: {aggs}')
                logging.debug(f'Selected preds: {preds}')
                self.cache(aggs, preds)
    
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
        return AggQuery.from_fact(
            self.table, self.all_preds, 
            self.cmp_pred, self.agg_cols, fact)
    
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