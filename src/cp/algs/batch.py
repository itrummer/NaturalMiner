'''
Created on Aug 15, 2021

@author: immanueltrummer
'''
import collections
import cp.algs.rl
import cp.cache.static
import cp.cache.multi
import cp.sql.query
import cp.text.fact
import gym
from gym import spaces
import logging
import numpy as np
import random
from sklearn.cluster import KMeans
from stable_baselines3 import A2C, PPO
import statistics

def batch_text():
    """ Generate text for a batch of items, using same summary template. 
    
    Args:
        
    """
    pass

def eval_solution(connection, batch, all_preds, solution):
    """ Evaluates solution to batch summarization problem.
    
    Args:
        connection: connection to database
        batch: description of test case
        all_preds: all predicates on dimension columns
        solution: maps each predicate to a summary template
    
    Returns:
        dictionary mapping predicates to summaries with reward
    """
    table = batch['general']['table']
    preamble = batch['general']['preamble']
    dim_cols = batch['general']['dim_cols']
    dims_tmp = batch['general']['dims_tmp']
    agg_cols = batch['general']['agg_cols']
    aggs_txt = batch['general']['aggs_txt']
    cmp_col = batch['general']['cmp_col']
    
    cache = cp.cache.multi.MultiItemCache(connection, table, cmp_col, all_preds, agg_cols, sum_tmp)
    s_eval = cp.text.sum.SumEvaluator()
    
    result = {}
    for cmp_pred in batch['predicates']:
        
        q_engine = cp.sql.query.QueryEngine(
            connection, table, cmp_pred, cache)
        s_gen = cp.text.sum.SumGenerator(
            all_preds, preamble, dim_cols,
            dims_tmp, agg_cols, aggs_txt,
            q_engine)
        
        sum_tmp = solution[cmp_pred]
        sum_facts = [cp.text.fact.Fact.from_props(p) for p in sum_tmp]
        d_sum, _ = s_gen.generate(sum_facts)
        quality = s_eval.evaluate(d_sum)
        result[cmp_pred] = (d_sum, quality)
    
    return result

def simple_batch(connection, batch, all_preds):
    """ Simple baseline using same summary template for entire batch.
    
    Args:
        connection: connection to database
        batch: a batch of summarization tasks
        all_preds: all predicates on dimensions
    
    Returns:
        Dictionary mapping each predicate to summary template
    """
    test_case = batch['general'].copy()
    all_cmp_preds = batch['predicates']
    src_table = batch['general']['table']

    full_card = cp.sql.cost.cardinality(connection, src_table)
    sample_card = max(full_card * 0.01, 10000)
    sample = f'(select * from {src_table} limit {sample_card}) as T'
    test_case['table'] = sample
        
    cmp_preds = random.choices(all_cmp_preds, k=3)
    test_case['cmp_preds'] = cmp_preds

    env = cp.algs.rl.PickingEnv(
        connection, **test_case, all_preds=all_preds,
        c_type='empty', cluster=True)
    model = A2C(
        'MlpPolicy', env, verbose=True, 
        gamma=1.0, normalize_advantage=True)
    model.learn(total_timesteps=200)
    
    if env.props_to_rewards:
        best = sorted(env.props_to_rewards.items(), key=lambda i:i[1])[-1]
        best_props = best[0]
    else:
        nr_facts = test_case['nr_facts']
        nr_preds = test_case['nr_preds']
        fact = cp.text.fact.Fact(nr_preds)
        best_props = nr_facts * [fact]
    
    return {p:best_props for p in batch['predicates']}


class ClusterEnv(gym.Env):
    """ Environment teaching agent how to cluster items for summarization. """
    
    def __init__(self, connection, batch, all_preds, 
                 cmp_preds, nr_features, raw_features):
        """ Initializes weighting for given number of features.
        
        Args:
            connection: connection to database
            batch: batch of summarization tasks
            all_preds: all possible predicates on dimensions
            cmp_preds: predicates identifying items for comparison
            nr_features: number of features to weigh
            raw_features: cluster based on these features
        """
        self.connection = connection
        self.batch = batch
        self.all_preds = all_preds
        self.cmp_preds = cmp_preds
        self.nr_features = nr_features
        self.raw_features = raw_features
        self.nr_items = len(cmp_preds)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(nr_features,), 
            dtype=np.float32)
        self.observation_space = spaces.Discrete(1)
        self.best_clusters = None
        self.best_reward = float('-inf')
    
    def reset(self):
        return 0
        
    def step(self, action):
        """ Executes one step of reinforcement learning. 
        
        Args:
            action: new set of weights to try
        """
        weights = action
        clusters = self._cluster(weights)
        reward = self._eval_clusters(clusters)
        if reward > self.best_reward:
            self.best_clusters = clusters

        logging.debug(f'Reward {reward} for weights {weights}')
        for c_id, c_items in clusters.items():
            logging.debug(f'Cluster {c_id}: {c_items}')
        return 0, reward, False, {}
    
    def _cluster(self, weights):
        """ Clusters items using features weighted by given weights. 
        
        Args:
            weights: multiple distance for each feature by those weights
        
        Returns:
            item clusters (set of item sets)
        """
        w_features = []
        for cmp_pred in self.cmp_preds:
            cur_w_features = []
            cur_r_features = self.raw_features[cmp_pred]
            for f, w in zip(cur_r_features, weights):
                cur_w_features.append(f*w)
            w_features.append(cur_w_features)
        
        X = np.array(w_features)
        kmeans = KMeans(n_clusters=5).fit(X)
        
        clusters = collections.defaultdict(lambda:set())
        for item, label in enumerate(kmeans.labels_):
            clusters[label].add(self.cmp_preds[item])
        
        return clusters
    
    def _eval_clusters(self, clusters):
        """ Estimate average summary quality with given clusters.
        
        Args:
            clusters: maps cluster IDs to sets of comparison predicates

        Returns:
            estimated average quality when generating one summary per cluster
        """
        # select random cluster, probability weighted by cluster size
        item_idx = random.randint(0, self.nr_items-1)
        cmp_pred = self.cmp_preds[item_idx]
        c_id = [c for c, items in clusters.items() if cmp_pred in items][0]
        
        # generate summary and estimate generalization for items in cluster
        cluster_items = list(clusters[c_id])
        cluster_batch = self.batch.copy()
        cluster_batch['predicates'] = random.choices(cluster_items, k=5)
        
        solution = simple_batch(
            self.connection, cluster_batch, 
            self.all_preds)
        eval_sol = eval_solution(
            self.connection, cluster_batch, 
            self.all_preds, solution)

        mean = statistics.mean([r for (_, r) in eval_sol.values()])
        logging.debug(f'Mean {mean} for {eval_sol}')
        return mean

class BatchProcessor():
    """ Generates summaries for item clusters. """
    
    def __init__(self, connection, batch, all_preds):
        """ Initializes for given batch. 
        
        Args:
            connection: connection to database
            batch: batch of test cases
            all_preds: predicates on dimensions
        """
        self.connection = connection
        self.batch = batch
        self.all_preds = all_preds
        self.table = batch['general']['table']
        self.cmp_preds = batch['predicates']
        self.nr_features, self.raw_features = self._get_features()
        self.cluster_env = ClusterEnv(
            connection, batch, all_preds, 
            self.cmp_preds, self.nr_features, 
            self.raw_features)
    
    def summarize(self):
        """ Generates summaries for clusters of items.
        
        Returns:
            dictionary mapping cluster IDs to a solution (item->summary)
        """
        model = A2C(
            'MlpPolicy', self.cluster_env, verbose=True, 
            gamma=1.0, normalize_advantage=True)
        model.learn(total_timesteps=20)
        clusters = self.cluster_env.best_clusters
        logging.info(f'Clusters: {clusters}')
        
        result = {}
        for c_id, cmp_preds in clusters.items():
            logging.info(f'Processing cluster {cmp_preds}')
            cluster_batch = self.batch.copy()
            cluster_batch['predicates'] = list(cmp_preds)
            result[c_id] = simple_batch(
                self.connection, cluster_batch, self.all_preds)
        
        return result
    
    def _get_features(self):
        """ Collect raw features used to cluster items. 
        
        Returns:
            Nr. features, dictionary mapping items to feature vectors
        """
        results = {}
        agg_cols = self.batch['general']['agg_cols']
        dim_cols = self.batch['general']['dim_cols']
        
        with self.connection.cursor() as cursor:
            
            agg_features = [f'avg({a}) as {a}' for a in agg_cols]
            for agg_col in agg_cols:
                for dim_col in dim_cols:
                    sql = f'select {dim_col} as val from {self.table} limit 1'
                    cursor.execute(sql)
                    row = cursor.fetchone()
                    val = row['val']
                    agg_feature = f"(select sum(case when {dim_col} = '{val}' " \
                        f"then {agg_col} else 0 end))/(select count({agg_col}))"
                    agg_features.append(agg_feature)
            
            nr_features = len(agg_features)
            
            for cmp_pred in self.cmp_preds:
                sql = 'select ' + ', '.join(agg_features) + \
                    ' from (select * from ' + self.table + \
                    ' limit 10000) as T where ' + cmp_pred
                logging.debug(f'Feature query: {sql}')
                cursor.execute(sql)
                row = cursor.fetchone()
                result = [row[a] for a in agg_cols]
                result = [float(r) if r is not None else 0 for r in result]
                results[cmp_pred] = result
        
        logging.debug(results)
        return nr_features, results