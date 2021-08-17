'''
Created on Jun 5, 2021

@author: immanueltrummer
'''
from collections import defaultdict
from cp.sql.query import AggQuery
import cp.cache.dynamic
import cp.cache.static
import cp.cache.proactive
from cp.text.fact import Fact
from cp.sql.pred import is_pred
from cp.sql.query import QueryEngine
from cp.text.sum import SumGenerator, SumEvaluator
from gym import spaces
from sentence_transformers import SentenceTransformer, util
import gym
import logging
import numpy as np
import statistics
import torch

class EmbeddingGraph():
    """ Graph connecting nodes with similar label embeddings. """

    def __init__(self, labels, degree, cluster):
        """ Generate graph with given labels.
        
        Args:
            labels: text labels for nodes
            degree: number of neighbors per node
            cluster: whether to cluster nodes by embedding
        """
        model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
        self.embeddings = model.encode(labels, convert_to_tensor=True)
        cosine_sims = util.pytorch_cos_sim(self.embeddings, self.embeddings)
        self.neighbors = []
        
        nr_labels = len(labels)
        for i in range(nr_labels):
            prefs = cosine_sims[i,i:nr_labels]
            k = min(prefs.shape[0], degree)
            if cluster:
                _, indices = torch.topk(prefs, k)
                l_indices = indices.tolist()
            else:
                l_indices = list(range(i, i+k))
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
    """ Environment for selecting facts for data summaries. """
    
    def __init__(self, connection, table, dim_cols, 
                 agg_cols, cmp_preds, nr_facts, nr_preds,
                 degree, max_steps, preamble, dims_tmp, 
                 aggs_txt, all_preds, c_type, cluster):
        """ Read database to initialize environment. 
        
        Args:
            connection: connection to database
            table: name of table to extract facts
            dim_cols: names of all property columns
            agg_cols: name of columns for aggregation
            cmp_preds: list of comparison predicates
            nr_facts: at most that many facts in description
            nr_preds: at most that many predicates per fact
            degree: degree for all transition graphs
            max_steps: number of steps per episode
            preamble: starts each data summary
            dims_tmp: assigns each dimension to text template
            aggs_txt: assigns each aggregate to text snippet
            all_preds: all possible predicates
            c_type: type of cache to create
            cluster: whether to cluster search space by embedding
        """
        super(PickingEnv, self).__init__()
        self.connection = connection
        self.table = table
        self.dim_cols = dim_cols
        self.agg_cols = agg_cols
        self.cmp_preds = cmp_preds
        self.nr_items = len(cmp_preds)
        self.nr_facts = nr_facts
        self.nr_preds = nr_preds
        self.degree = degree
        self.max_steps = max_steps
        self.preamble = preamble
        self.dims_tmp = dims_tmp
        self.aggs_txt = aggs_txt
        
        self.agg_graph = EmbeddingGraph(agg_cols, degree, cluster)
        self.all_preds = all_preds
        pred_labels = [f'{p} is {v}' for p, v in self.all_preds]
        self.pred_graph = EmbeddingGraph(pred_labels, degree, cluster)
        
        self.proactive = True if c_type == 'proactive' else False
        self.caches = []
        self.q_engines = []
        self.s_gens = []
        for cmp_pred in cmp_preds:
            cache = self._create_cache(
                connection, table, dim_cols, 
                cmp_pred, agg_cols, nr_facts, 
                nr_preds, all_preds, c_type)
            q_engine = QueryEngine(
                connection, table, 
                cmp_pred, cache)
            s_gen = SumGenerator(
                all_preds, preamble, dim_cols, 
                dims_tmp, agg_cols, aggs_txt, 
                q_engine)
            self.caches.append(cache)
            self.q_engines.append(q_engine)
            self.s_gens.append(s_gen)

        self.s_eval = SumEvaluator()
        self.props_to_rewards = {}
        self.props_to_conf = {}
        
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
        stats = defaultdict(lambda:0)
        stats.update(self.s_eval.statistics())
        for cache in self.caches:
            stats.update(cache.statistics())
        for engine in self.q_engines:
            stats.update(engine.statistics())
        for gen in self.s_gens:
            stats.update(gen.statistics())
        
        return stats

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
            for cache in self.caches:
                if self.proactive:
                    cache.set_cur_facts(self.cur_facts)
                cache.update()
            reward = self._evaluate()
        
        return self._observe(), reward, done, {}

    def _create_cache(self, connection, table, dim_cols, cmp_pred, 
                      agg_cols, nr_facts, nr_preds, all_preds, c_type):
        """ Creates a cache object as specified.
        
        Args:
            connection: connection to database
            table: cache stores results of queries on this table
            dim_cols: dimension columns
            cmp_pred: compare data subsets defined by this predicate
            agg_cols: aggregation columns
            nr_facts: summary contains so many facts
            nr_preds: maximal number of predicates per fact
            all_preds: all possible predicates on dimensions
            c_type: string identifier of cache type
        Returns:
            newly created cache object
        """
        if c_type == 'dynamic':
            cache = cp.cache.dynamic.DynamicCache(connection)
        elif c_type == 'empty':
            cache = cp.cache.static.EmptyCache()
        elif c_type == 'cube':
            cache = cp.cache.static.CubeCache(
                connection, table, dim_cols, cmp_pred, agg_cols, 900)
        elif c_type == 'proactive':
            cache = cp.cache.proactive.ProCache(
                connection, table, cmp_pred, nr_facts, 
                nr_preds, all_preds, self.pred_graph,
                agg_cols, self.agg_graph)
        else:
            raise ValueError(f'Unknown cache type: {c_type}')
        return cache

    def _evaluate(self):
        """ Evaluate quality of current summary. """
        confs = []
        rewards = []
        for gen in self.s_gens:
            text, conf = gen.generate(self.cur_facts)
            reward = self.s_eval.evaluate(text)
            rewards.append(reward)
            if conf is not None:
                confs.append(conf)
        
        mean_conf = statistics.mean(confs) if confs else 0
        mean_reward = statistics.mean(rewards)
        self._save_eval_results(mean_reward, mean_conf)
        return mean_reward

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
    
    def _save_eval_results(self, reward, conf):
        """ Store reward of current fact combination. 
        
        Args:
            reward: received for current facts
            conf: confidence of evaluation (used if sampling)
        """
        fact_tuples = [tuple(f.props) for f in self.cur_facts]
        sum_tuple = tuple(fact_tuples)
        self.props_to_rewards[sum_tuple] = reward
        self.props_to_conf[sum_tuple] = conf