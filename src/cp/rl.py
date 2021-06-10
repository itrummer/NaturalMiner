'''
Created on Jun 5, 2021

@author: immanueltrummer
'''
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import cp.query
import gym
from gym import spaces
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
            _, indices = torch.topk(cosine_sims[i], degree)
            self.neighbors.append(indices)

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


class Fact():
    """ Represents one fact in a data summary. """
    
    def __init__(self, nr_preds):
        """ Initialize fact for given number of predicates. 
        
        Args:
            nr_preds: (maximal) number of scope predicates
        """
        self.nr_preds = nr_preds
        self.nr_props = nr_preds + 1
        
    def get_agg(self):
        """ Returns index of aggregate. """
        return self.props[self.nr_preds]
        
    def get_preds(self):
        """ Returns fact predicates (indices). """
        return self.props[0:self.nr_preds]
        
    def change(self, prop_id, new_val):
        """ Change fact property to new value. 
        
        Args:
            prop_id: change value of this property
            new_val: assign property to this value
        """
        self.props[prop_id] = new_val
            
    def reset(self):
        """ Reset properties to default values. """
        self.props = [0] * self.nr_props


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
        self.q_engine = cp.query.QueryEngine(connection, table, cmp_pred)
        self.judge = pipeline("sentiment-analysis", 
                              model="siebert/sentiment-roberta-large-english")
        
        self.agg_graph = EmbeddingGraph(agg_cols, degree)
        self.all_preds = self._preds()
        pred_labels = [f'{p} is {v}' for p, v in self.all_preds]
        self.pred_graph = EmbeddingGraph(pred_labels, degree)

        self.cur_facts = []
        for _ in range(nr_facts):
            self.cur_facts.append(Fact(nr_preds))
        self.text_to_reward = {}

        self.props_per_fact = nr_preds + 1
        action_dims = [nr_facts + 1, self.props_per_fact, degree]
        self.action_space = spaces.MultiDiscrete(action_dims)
        
        self.nr_props = nr_facts * self.props_per_fact
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(self.nr_props, 384), dtype=np.float32)
    
    def best_summary(self):
        """ Returns data summary with highest reward. """
        return max(self.text_to_reward, key=self.text_to_reward.get)
    
    def step(self, action):
        """ Change fact or trigger evaluation. """
        self.nr_steps += 1
        
        if self.nr_steps >= 100 or action[0] >= self.nr_facts:
            done = True
            reward = self._evaluate()
        else:
            fact_idx = action[0]
            prop_idx = action[1]
            new_val = action[2]
            self.cur_facts[fact_idx].change(prop_idx, new_val)
            
            done = False
            reward = 0
        
        return self._observe(), reward, done, {}
        
    def reset(self):
        """ Reset data summary to default. """
        self.nr_steps = 0
        for fact in self.cur_facts:
            fact.reset()
            
        return self._observe()
        
    def _evaluate(self):
        """ Evaluate quality of current summary. """
        text = self._generate()
        sent = self.judge(text)[0]
        label = sent['label']
        score = sent['score']
        
        if label == 'POSITIVE':
            reward = score
        else:
            reward = -score
        print(f'Reward {reward} for "{text}"')
        self.text_to_reward[text] = reward
        return reward
    
    def _generate(self):
        """ Generate textual data summary. """
        s_parts = []
        for fact in self.cur_facts:
            s_parts.append(f'Among all {self.table} ')
            preds = [self.all_preds[i] for i in fact.get_preds()]
            for pred in preds:
                s_parts.append(f'with {pred[0]} {pred[1]}, ')
                
            agg_idx = fact.get_agg()
            agg_col = self.agg_cols[agg_idx]
            rel_avg = self.q_engine.rel_avg(preds, agg_col)
            s_parts.append(f'its average {agg_col} is {rel_avg} times the default.')
        
        return ' '.join(s_parts)
        
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
        
        return torch.stack(components, dim=0).numpy()
        
    def _preds(self):
        """ Generates all possible equality predicates. 
        
        Returns:
            list of (column, value) pairs representing predicates
        """
        preds = []
        for dim in self.dim_cols:
            with self.connection.cursor() as cursor:
                query = f'select distinct {dim} from {self.table} ' \
                    f'where {self.cmp_pred}'
                cursor.execute(query)
                result = cursor.fetchall()
                preds += [(dim, r[0]) for r in result]
        return preds