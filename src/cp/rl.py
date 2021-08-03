'''
Created on Jun 5, 2021

@author: immanueltrummer
'''
from collections import defaultdict
from cp.cache.common import AggQuery
from cp.cache.dynamic import View
from cp.fact import Fact
from cp.pred import is_pred
from cp.query import QueryEngine
from cp.sum import SumGenerator, SumEvaluator
from gym import spaces
from sentence_transformers import SentenceTransformer, util
import gym
import logging
import numpy as np
import torch

class EmbeddingGraph():
    """ Graph connecting nodes with similar label embeddings. """
    
    def __init__(self, labels, degree):
        """ Generate graph with given labels.
        
        Args:
            labels: text labels for nodes
            degree: number of neighbors per node
        """
        model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
        self.embeddings = model.encode(labels, convert_to_tensor=True)
        cosine_sims = util.pytorch_cos_sim(self.embeddings, self.embeddings)
        self.neighbors = []
        
        nr_labels = len(labels)
        for i in range(nr_labels):
            prefs = cosine_sims[i,i:nr_labels]
            k = min(prefs.shape[0], degree)
            _, indices = torch.topk(prefs, k)
            l_indices = indices.tolist()
            l_indices += [0] * (degree - k)
            self.neighbors.append(l_indices)

    def get_embedding(self, node_id):
        """ Returns embedding of given node.
        
        Args:
            node_id: retrieve embedding for this node
            
        Returns:
            transformer embedding for given node
        """
        return self.embeddings[node_id]

    def get_neighbor(self, node, n_id):
        """ Retrieve specific neighbor of node. 
        
        Args:
            node: ID of node whose neighbors to examine
            n_id: index of node neighbor to retrieve
            
        Returns:
            i-th neighbor of j-th node
        """
        return self.neighbors[node][n_id]
    
    def get_reachable(self, start, steps):
        """ Retrieve all nodes reachable within a given number of steps. 
        
        Args:
            start: ID of start node
            steps: maximal number of steps
            
        Returns:
            all nodes reachable within given number of steps
        """
        reachable = set([start])
        for _ in range(steps):
            boundary = [nb for n in reachable for nb in self.neighbors[n]]
            reachable.update(boundary)
            
        return reachable

class PickingEnv(gym.Env):
    """ Environment for selecting facts for a data summary. """
    
    def __init__(self, connection, table, dim_cols, 
                 agg_cols, cmp_pred, nr_facts, nr_preds,
                 degree, max_steps, preamble, dims_tmp, 
                 aggs_txt, all_preds, cache, proactive):
        """ Read database to initialize environment. 
        
        Args:
            connection: connection to database
            table: name of table to extract facts
            dim_cols: names of all property columns
            agg_cols: name of columns for aggregation
            cmp_pred: compare data satisfying this predicate
            nr_facts: at most that many facts in description
            nr_preds: at most that many predicates per fact
            degree: degree for all transition graphs
            max_steps: number of steps per episode
            preamble: starts each data summary
            dims_tmp: assigns each dimension to text template
            aggs_txt: assigns each aggregate to text snippet
            all_preds: all possible predicates
            cache: caches query results
            proactive: whether to cache proactively
        """
        super(PickingEnv, self).__init__()
        self.connection = connection
        self.table = table
        self.dim_cols = dim_cols
        self.agg_cols = agg_cols
        self.cmp_pred = cmp_pred
        self.nr_facts = nr_facts
        self.nr_preds = nr_preds
        self.degree = degree
        self.max_steps = max_steps
        self.preamble = preamble
        self.dims_tmp = dims_tmp
        self.aggs_txt = aggs_txt
        self.cache = cache
        self.proactive = proactive
        self.q_engine = QueryEngine(
            connection, table, cmp_pred, cache)
        self.s_gen = SumGenerator(
            all_preds, preamble, dim_cols, 
            dims_tmp, agg_cols, aggs_txt, 
            self.q_engine)
        self.s_eval = SumEvaluator()
        
        self.agg_graph = EmbeddingGraph(agg_cols, degree)
        self.all_preds = all_preds
        pred_labels = [f'{p} is {v}' for p, v in self.all_preds]
        self.pred_graph = EmbeddingGraph(pred_labels, degree)

        self.cur_facts = []
        for _ in range(nr_facts):
            self.cur_facts.append(Fact(nr_preds))

        self.props_per_fact = nr_preds + 1
        action_dims = [nr_facts, self.props_per_fact, degree]
        self.action_space = spaces.MultiDiscrete(action_dims)
        
        self.nr_props = nr_facts * self.props_per_fact
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(self.nr_props, 384), dtype=np.float32)
        
        self.eval_s = 0
        self.nr_t_steps = 0
        self.reset()
        self._evaluate()
    
    def best_summary(self):
        """ Returns data summary with highest reward. """
        return max(self.text_to_reward, key=self.text_to_reward.get)
    
    def topk_summaries(self, k, best):
        """ Returns the top-k data summaries.
        
        Args:
            k: retrieve that many summaries
            best: return best (not worst) summaries
            
        Returns:
            returns up to k summaries with rewards
        """
        sorted_sums = sorted(self.text_to_reward.items(), 
                             key=lambda s: s[1], reverse=best)
        if len(self.text_to_reward) < k:
            return sorted_sums
        else:
            return sorted_sums[0:k]

    def step(self, action):
        """ Change fact or trigger evaluation. """
        self.nr_ep_steps += 1
        self.nr_t_steps += 1
        logging.debug(f'Step {self.nr_t_steps} (in episode: {self.nr_ep_steps})')
        
        if self.nr_ep_steps >= self.max_steps:
            done = True
            reward = 0
        else:
            fact_idx = action[0]
            prop_idx = action[1]
            nb_idx = action[2]
            fact = self.cur_facts[fact_idx]
            cur_val = fact.get_prop(prop_idx)
            if fact.is_agg(prop_idx):
                new_val = self.agg_graph.get_neighbor(cur_val, nb_idx)
            else:
                new_val = self.pred_graph.get_neighbor(cur_val, nb_idx)
            self.cur_facts[fact_idx].change(prop_idx, new_val)
            
            done = False
            if self.proactive:
                self._proactive_caching()
            reward = self._evaluate()
        
        return self._observe(), reward, done, {}
        
    def reset(self):
        """ Reset data summary to default. """
        self.nr_ep_steps = 0
        for fact in self.cur_facts:
            fact.reset()
            
        return self._observe()
    
    def statistics(self):
        """ Returns performance statistics. 
        
        Returns:
            Dictionary with performance statistics
        """
        stats = self.q_engine.statistics().copy()
        stats.update(self.s_gen.statistics())
        stats.update(self.s_eval.statistics())
        return stats
    
    def _evaluate(self):
        """ Evaluate quality of current summary. """
        text = self.s_gen.generate(self.cur_facts)
        return self.s_eval.evaluate(text)

    def _observe(self):
        """ Returns observations for learning agent. """
        components = []
        for fact_ctr in range(self.nr_facts):
            fact = self.cur_facts[fact_ctr]
            preds = fact.get_preds()
            for pred_idx in preds:
                pred_emb = self.pred_graph.get_embedding(pred_idx)
                components.append(pred_emb)
                
            agg_idx = fact.get_agg()
            agg_emb = self.agg_graph.get_embedding(agg_idx)
            components.append(agg_emb)
        
        return torch.stack(components, dim=0).to('cpu').numpy()
    
    def _proactive_caching(self):
        """ Fill cache with values for current facts and beyond. """
        
        # collect un-cached fact queries
        u_queries = []
        u_preds = set()
        u_aggs = set()
        for fact in self.cur_facts:
            eq_preds = [self.all_preds[p] for p in fact.get_preds()]
            eq_preds = list(filter(lambda p: is_pred(p), eq_preds))
            agg_idx = fact.get_agg()
            agg_col = self.agg_cols[agg_idx]
            query = AggQuery(self.table, eq_preds, self.cmp_pred, agg_col)
            if not self.cache.can_answer(query):
                u_preds.update(fact.get_preds())
                u_aggs.update([fact.get_agg()])
                u_queries.append(query)
                        
        # calculate aggregate scope
        rel_agg_ids = set()
        for u_agg in u_aggs:
            reachable = self.agg_graph.get_reachable(u_agg, 1)
            rel_agg_ids.update(reachable)
        rel_aggs = frozenset([self.agg_cols[a_id] for a_id in rel_agg_ids])
        if not rel_aggs:
            return
        
        # calculate predicate scope
        rel_dims = frozenset([p[0] for q in u_queries for p in q.eq_preds])
        d_to_v = defaultdict(lambda:set())
        for p in u_preds:
            for p_idx in self.pred_graph.get_reachable(p, 1):
                dim, val = self.all_preds[p_idx]
                if dim in rel_dims:
                    d_to_v[dim].add(val)
        
        # prune out cached results
        if len(d_to_v) == 1:
            logging.debug(f'Unpruned values: {d_to_v}')
            d, vals = list(d_to_v.items())[0]
            vals = self._prune_vals(d, vals, rel_aggs)
            logging.debug(f'Pruned values: {vals}')
            scope = frozenset([(d, frozenset(vals))])
        else:
            scope = frozenset([(d, frozenset(v)) for d, v in d_to_v.items()])

        # construct view for caching
        v = View(self.table, rel_dims, self.cmp_pred, rel_aggs, scope)
        logging.debug(f'Proactive caching selects {v}')
        self.cache.put_results(v)
        
    def _prune_vals(self, dim, vals, aggs):
        """ Prune out values for which all aggregates are cached.
        
        Args:
            dim: dimension column
            vals: values in dimension column to prune
            aggs: relevant aggregation columns
        
        Returns:
            subset of values for which not all aggregates are cached
        """
        keep_vals = []
        for val in vals:
            cached = True
            for agg in aggs:
                eq_preds = frozenset([(dim, val)])
                query = AggQuery(self.table, eq_preds, self.cmp_pred, agg)
                if not self.cache.can_answer(query):
                    cached = False
                    break
            
            if not cached:
                keep_vals += [val]
        
        return keep_vals