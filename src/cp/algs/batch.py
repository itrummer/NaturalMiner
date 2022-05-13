'''
Created on Aug 15, 2021

@author: immanueltrummer
'''
import collections
import cp.algs.rl
import cp.algs.sample
import cp.cache.multi
import cp.sql.query
import cp.text.fact
import gym
from gym import spaces
import logging
import numpy as np
import random
from sklearn.cluster import KMeans
from stable_baselines3 import A2C
import statistics
from cp.sql.pred import pred_sql


def eval_solution(connection, batch, all_preds, solution):
    """ Evaluates solution to batch summarization problem.
    
    Careful: this function assumes that comparison predicates
    are unary equality predicates on the comparison column.
    If this is not the case, it adds unnecessary overheads
    (the result is still correct). Might refine later.
    
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
    cmp_col = batch['cmp_col']
    
    # collect items with same summary template
    sum_to_preds = collections.defaultdict(lambda:[])
    for pred, sum_tmp in solution.items():
        sum_to_preds[sum_tmp] += [pred]
    
    result = {}
    for sum_tmp, cmp_preds in sum_to_preds.items():
        
        cache = cp.cache.multi.MultiItemCache(
            connection, table, cmp_col, 
            all_preds, agg_cols, sum_tmp)
        s_eval = cp.text.sum.SumEvaluator()
        
        logging.info('Generating text summaries ...')
        cmp_sums = []
        for cmp_pred in cmp_preds:
            q_engine = cp.sql.query.QueryEngine(
                connection, table, cmp_pred, cache)
            s_gen = cp.text.sum.SumGenerator(
                all_preds, preamble, dim_cols,
                dims_tmp, agg_cols, aggs_txt,
                q_engine)
            sum_tmp = solution[cmp_pred]
            sum_facts = [cp.text.fact.Fact.from_props(p) for p in sum_tmp]
            d_sum, _ = s_gen.generate(sum_facts)
            cmp_sums += [(cmp_pred, d_sum)]

        logging.info('Evaluating summary quality ...')
        for cmp_pred, d_sum in cmp_sums:
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
    if not all_cmp_preds:
        return {}

    to_select = min(3, len(all_cmp_preds))
    cmp_preds = random.choices(all_cmp_preds, k=to_select)
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


class IterativeClusters():
    """ Iteratively divides items into clusters to generate summaries. """
    
    def __init__(self, connection, batch, all_preds):
        """ Initialize for given problem.
        
        Args:
            connection: connection to database
            batch: batch containing all items
            all_preds: all predicates on dimensions
        """
        self.connection = connection
        self.all_preds = all_preds
        solution = simple_batch(connection, batch, all_preds)
        s_eval = eval_solution(connection, batch, all_preds, solution)
        self.id_to_be = {0:(batch, s_eval)}

    def iterate(self):
        """ Performs one iteration. """
        logging.debug(f'Batches before iteration: {self.id_to_be}')
        to_split_idx = self._select()
        to_split = self.id_to_be[to_split_idx]
        split = self._even_split(to_split)
        del self.id_to_be[to_split_idx]
        
        for b in split:
            solution = simple_batch(self.connection, b, self.all_preds)
            s_eval = eval_solution(self.connection, b, self.all_preds, solution)
            b_id = self._next_ID()
            self.id_to_be[b_id] = b, s_eval
        
        for b_id, (b, b_eval) in self.id_to_be.items():
            logging.debug(f'Cluster {b_id}: {b}, {b_eval}')

    def _even_split(self, b_e):
        """ Splits items, sorted by text quality, evenly into two halves.
        
        Args:
            b_e: batch with associated evaluation
        
        Returns:
            list of two batches splitting the input batch
        """
        batch, eval_s = b_e
        nr_preds = len(eval_s)
        preds_by_qual = sorted(eval_s.keys(), key=lambda p:eval_s[p][1])
                
        split_batches = []
        middle = round(nr_preds/2)
        for s in [slice(0, middle), slice(middle, nr_preds)]:
            cmp_preds = preds_by_qual[s]
            split_batch = batch.copy()
            split_batch['predicates'] = cmp_preds
            split_batches.append(split_batch)
        
        return split_batches

    def _next_ID(self):
        """ Returns next available batch ID.
        
        Returns:
            integer representing next batch ID
        """
        ids = self.id_to_be.keys()
        return 1+max(ids) if ids else 0

    def _priority_avg(self, b_e):
        """ Evaluates priority for splitting batch using average quality.
        
        Args:
            b_e: batch with associated evaluation
        
        Returns:
            splitting priority (high priority means likely split)
        """
        _, s_eval = b_e
        # bad_items = [i for i, (_, q) in s_eval.items() if q < 0]
        # return len(bad_items)
        return - statistics.mean([v[1] for v in s_eval.values()])
    
    def _priority_count(self, b_e):
        """ Evaluates priority for splitting batch using count of bad items.
        
        Args:
            b_e: batch with associated evaluation
        
        Returns:
            splitting priority (high priority means likely split)
        """
        _, s_eval = b_e
        bad_items = [i for i, (_, q) in s_eval.items() if q < 0]
        return len(bad_items)

    def _select(self):
        """ Select one of current batches to split. 
        
        Returns:
            index of batch to split
        """
        ids = self.id_to_be.keys()
        return max(ids, key=lambda b_id:
                   self._priority_avg(
                       self.id_to_be[b_id]))
    
    def _signum_split(self, b_e):
        """ Splits items based on the signum of text quality.
        
        Args:
            b_e: batch with associated evaluation
        
        Returns:
            list of two batches splitting the input batch
        """
        batch, eval_s = b_e
        preds_1 = [p for p, (_, q) in eval_s.items() if q < 0]
        preds_2 = [p for p, (_, q) in eval_s.items() if q >= 0]
        
        split_batches = []
        for preds in [preds_1, preds_2]:
            split_batch = batch.copy()
            split_batch['predicates'] = preds
            split_batches.append(split_batch)
        
        return split_batches


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
        model.learn(total_timesteps=10)
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
        
        Careful: this implementation assumes that comparison
        predicates are unary equality predicates on the same
        column (therefore, all features can be retrieved by
        a single group-by query).
        
        Returns:
            Nr. features, dictionary mapping items to feature vectors
        """
        results = {}
        agg_cols = self.batch['general']['agg_cols']
        dim_cols = self.batch['general']['dim_cols']
        cmp_col = self.batch['cmp_col']
        
        with self.connection.cursor() as cursor:
            
            agg_features = [f'avg({a}) as {a}' for a in agg_cols]
            # for agg_col in agg_cols:
                # for dim_col in dim_cols:
                    # sql = f'select {dim_col} as val from {self.table} limit 1'
                    # cursor.execute(sql)
                    # row = cursor.fetchone()
                    # val = row['val']
                    # e_val = cp.sql.pred.sql_esc(val)
                    # agg_feature = \
                        # f"(select sum(case when {dim_col} = '{e_val}' " \
                        # f"then {agg_col} else 0 end))/(select count({agg_col}))"
                    # agg_features.append(agg_feature)
            
            nr_features = len(agg_features)
            sql = 'select ' + ', '.join(agg_features) + \
                f', {cmp_col} as cmp_val ' + \
                f'from {self.table} group by {cmp_col}'
            logging.debug(f'Feature query: {sql}')
            cursor.execute(sql)
            rows = cursor.fetchall()
            for row in rows:
                result = [row[a] for a in agg_cols]
                result = [float(r) if r is not None else 0 for r in result]
                cmp_val = row['cmp_val']
                cmp_pred = pred_sql(cmp_col, cmp_val)
                results[cmp_pred] = result
        
        logging.debug(results)
        return nr_features, results


class SubModularIterative():
    """ Iteratively improves summaries based on sub-modularity. """
    
    def __init__(self, connection, batch, all_preds):
        """ Initialize for given problem.
        
        Args:
            connection: connection to database
            batch: batch containing all items
            all_preds: all predicates on dimensions
        """
        self.connection = connection
        self.batch = batch
        self.all_preds = all_preds
        self.all_cmp_preds = batch['predicates']
        self.best_sums = {p:('', -1) for p in self.all_cmp_preds}

    def iterate(self):
        """ Performs one iteration to improve summary quality. 
        
        The algorithm selects a sample of items. Next, it uses
        reinforcement learning to find a summary sketch which
        improves summary quality for those items maximally.
        Each item is assigned to the summary with maximal 
        quality (either the new summary or a prior summary).
        """
        logging.info('Picking new sketch ...')
        best_props = self._pick_sketch()
        solution = {
            cmp_pred:best_props for 
            cmp_pred in self.all_cmp_preds}
        logging.info('Evaluating sketch ...')
        s_eval = eval_solution(
            self.connection, self.batch, 
            self.all_preds, solution)
        logging.info('Pruning sketches ...')
        for cmp_pred, (d_sum, quality) in s_eval.items():
            _, prior_quality = self.best_sums[cmp_pred]
            if prior_quality < quality:
                self.best_sums[cmp_pred] = d_sum, quality
    
    def _pick_sketch(self):
        """ Pick a summary sketch that is optimally complementary.
        
        Picks a summary sketch that optimally complements previously
        considered summary sketches. This means that the improvement
        by assigning items not covered well by previous summaries
        to this sketch is maximal.
        
        Returns:
            List of facts representing summary sketch
        """
        print('Selecting next sketch ...')
        test_case = self.batch['general'].copy()
        all_cmp_preds = self.batch['predicates']
        if not all_cmp_preds:
            return {}
    
        to_select = min(10, len(all_cmp_preds))
        cmp_preds = random.choices(all_cmp_preds, k=to_select)
        test_case['cmp_preds'] = cmp_preds
        
        table = test_case['table']
        sample_ratio = 0.1
        logging.info(f'Sampling {table} with ratio {sample_ratio} ...')
        sample_table = cp.algs.sample.create_sample(
            self.connection, table, sample_ratio)
        test_case['table'] = sample_table
    
        total_timesteps = 50
        logging.info(f'Iterating for {total_timesteps} steps ...')
        env = cp.algs.rl.PickingEnv(
            self.connection, **test_case, 
            all_preds=self.all_preds,
            c_type='proactive', cluster=True,
            prior_best=self.best_sums)
        model = A2C(
            'MlpPolicy', env, verbose=True, 
            gamma=1.0, normalize_advantage=True)
        model.learn(total_timesteps=total_timesteps)
        
        if env.props_to_rewards:
            best = sorted(env.props_to_rewards.items(), key=lambda i:i[1])[-1]
            best_props = best[0]
        else:
            nr_facts = test_case['nr_facts']
            nr_preds = test_case['nr_preds']
            fact = cp.text.fact.Fact(nr_preds)
            best_props = nr_facts * [fact]
        
        return best_props