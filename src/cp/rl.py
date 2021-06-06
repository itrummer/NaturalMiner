'''
Created on Jun 5, 2021

@author: immanueltrummer
'''
from sentence_transformers import SentenceTransformer, util
import gym
from gym import spaces
import numpy as np
import torch
from numpy import dtype

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
            _, indices = torch.topk(cosine_sims[i], degree)
            self.neighbors.append[indices]

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


class PickingEnv(gym.Env):
    """ Environment for selecting facts for a data summary. """
    
    def __init__(self, connection, table, dim_cols, 
                 agg_cols, cmp_pred, nr_facts, nr_preds,
                 degree):
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
        
        self.agg_graph = EmbeddingGraph(agg_cols, degree)
        self.all_preds = self._preds()
        pred_labels = [f'{p} is {v}' for p, v in self.all_preds]
        self.pred_graph = EmbeddingGraph(pred_labels, degree)

        self.nr_props = nr_facts * (nr_preds + 1)
        action_dims = [self.nr_props + 1, degree]
        self.action_space = spaces.MultiDiscrete(action_dims)
        
        self.cur_preds = [0] * nr_facts * nr_preds
        self.cur_aggs = [0] * nr_facts
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(self.nr_props, 768), dtype=np.float32)
    
    def step(self, action):
        self.nr_steps += 1
        pass
        
    def reset(self):
        """ Reset data summary to default. """
        self.nr_steps = 0
        for i in range(self.nr_facts * self.nr_):
            self.[i] = 0
        return self._observe()
        
    def _evaluate(self):
        """ Evaluate quality of current summary. """
        pass
        
    def _observe(self):
        """ Returns observations for learning agent. """
        components = []
        for fact_ctr in range(self.nr_facts):
            for pred_ctr in range(self.nr_preds):
                sum_idx = fact_ctr * (self.nr_preds+1) + pred_ctr
                pred_idx = self.summary[sum_idx]
                pred_emb = self.pred_graph.get_embedding(pred_idx)
                components.append(pred_emb)
                
            sum_idx = (fact_ctr+1) * (self.nr_preds+1) - 1
            agg_idx = self.summary[sum_idx]
            agg_emb = self.agg_graph.get_embedding(agg_idx)
            components.append(agg_emb)
        
        return torch.stack(components, dim=0)
        
    def _preds(self):
        """ Generates all possible equality predicates. 
        
        Returns:
            list of (column, value) pairs representing predicates
        """
        preds = []
        for dim in self.dim_cols:
            with self.connection.cursor() as cursor:
                query = f'select distinct {dim} from {self.table}'
                result = cursor.execute(query).fetchall()
                preds += [(dim, r[0]) for r in result]