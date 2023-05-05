'''
Created on Aug 7, 2021

@author: immanueltrummer
'''
from cp.sql.query import AggQuery, GroupQuery
import nminer.cache.dynamic
import nminer.text.fact
import nminer.algs.rl
import nminer.sql.cost
import nminer.text.sum
import logging
from stable_baselines3 import A2C
import time

def create_sample(connection, table, sample_ratio):
    """ Creates table containing sample from source.
    
    Args:
        connection: connection to database
        table: name of sample source table
        sample_ratio: ratio of samples
    
    Returns:
        name of table containing sample
    """
    total_rows, _ = nminer.sql.cost.estimates(
        connection, f'select * from {table}')
    sample_rows = int(
        max(10000, total_rows * sample_ratio))
    
    sample_tbl = f'{table}_sample'
    with connection.cursor() as cursor:
        cursor.execute(f'drop table if exists {sample_tbl}')
        cursor.execute(f'create unlogged table {sample_tbl} as ' \
                       f'(select * from {table} limit {sample_rows})')
    
    return sample_tbl


class Sampler():
    """ Selects data summaries via sampling. """
    
    def __init__(self, connection, test_case, all_preds, 
                 sample_ratio, max_nr_sums, c_type):
        """ Initialize sampler, creates summary generator.
        
        Args:
            connection: connection to database
            test_case: description of scenario
            all_preds: all possible predicates
            sample_ratio: sample at least this ratio of the rows
            max_nr_sums: maximal number of summaries on full data set
            c_type: cache type used for sampling
        """
        self.connection = connection
        self.test_case = test_case
        self.all_preds = all_preds
        self.sample_ratio = sample_ratio
        self.max_nr_sums = max_nr_sums
        self.c_type = c_type
        self.table = test_case['table']
        self.cmp_pred = test_case['cmp_pred']
        self.dim_cols = test_case['dim_cols']
        self.agg_cols = test_case['agg_cols']
        preamble = test_case['preamble']
        dims_tmp = test_case['dims_tmp']
        aggs_txt = test_case['aggs_txt']
        
        self.cache = nminer.cache.dynamic.DynamicCache(
            connection, self.table, self.cmp_pred)
        self.q_engine = nminer.sql.query.QueryEngine(
            connection, self.table, 
            self.cmp_pred, self.cache)
        self.s_gen = nminer.text.sum.SumGenerator(
            all_preds, preamble, self.dim_cols,
            dims_tmp, self.agg_cols, aggs_txt,
            self.q_engine)
        self.s_eval = nminer.text.sum.SumEvaluator()

    def _cost(self, g_queries):
        """ Calculates cost estimate for a set of group-by queries.
        
        Args:
            g_queries: group-by queries
        
        Returns:
            accumulated cost for set of group-by queries
        """
        total_cost = 0
        for g_q in g_queries:
            sql = g_q.sql()
            _, cost = nminer.sql.cost.estimates(self.connection, sql)
            total_cost += cost
        
        return total_cost

    
    def _fill_cache(self, sam_sums):
        """ Fill cache with query results needed to generate summaries.
        
        Args:
            sam_sums: summaries (facts) selected via sampling
        
        Returns:
            statistics on efficiency of merge step
        """
        logging.debug(f'Filling cache for summaries: {sam_sums}')
        stats = {}
        # collect queries associated with facts
        queries = []
        for sum_facts in sam_sums:
            for fact in sum_facts:
                a_q = AggQuery.from_fact(
                    self.table, self.all_preds,
                    self.cmp_pred, self.agg_cols, fact)
                queries += [a_q]
        
        logging.debug(f'Raw queries to cache: {queries}')
        
        # merge queries with same aggregates and dimensions
        g_merged = []
        for a_query in queries:
            is_merged = False
            for merged in g_merged:
                if merged.can_merge(a_query) and \
                a_query.agg_col in merged.agg_cols:
                    merged.integrate(a_query)
                    is_merged = True
                    break
            
            if not is_merged:
                g_query = GroupQuery.from_query(a_query)
                g_merged += [g_query]

        # merge queries with same dimensions until cost increases
        min_g_qs = g_merged.copy()
        min_cost = self._cost(g_merged)
        logging.debug(f'Cost before clustering: {min_cost} with {g_merged}')
        stats['no_merge_cost'] = min_cost
        
        improved = True
        while improved:
            sims = []
            for g1 in g_merged:
                for g2 in g_merged:
                    if not (g1 == g2):
                        sim = self._similarity(g1, g2)
                        sims += [(g1, g2, sim)]
            
            improved = False
            if sims:
                g1, g2, _ = max(sims, key=lambda s:s[2])
                gm = GroupQuery(self.table, g1.dims, self.cmp_pred)
                gm.preds.update(g1.preds)
                gm.preds.update(g2.preds)
                gm.aggs.update(g1.aggs)
                gm.aggs.update(g2.aggs)
                g_merged.remove(g1)
                g_merged.remove(g2)
                g_merged.append(gm)
                
                cost = self._cost(g_merged)
                if cost < min_cost:
                    min_cost = cost
                    min_g_qs = g_merged.copy()
                    improved = True
        
        logging.debug(f'Cost after clustering: {min_cost} with {min_g_qs}')
        stats['merged_cost'] = min_cost
        for g_q in min_g_qs:
            self.cache.cache(g_q)
        
        return stats
    
    def _full_data_summaries(self, sam_sums):
        """ Generates summaries by considering the full data set.
        
        Args:
            sam_sums: summaries selected via sample
        
        Returns:
            a subset of summaries referring to full data set, statistics
        """
        text_to_reward = {}
    
        nr_sums = min(len(sam_sums), self.max_nr_sums)
        sel_sums = sam_sums[0:nr_sums]
        if sel_sums:
            stats = self._fill_cache(sel_sums)
        else:
            stats = {}
        
        for facts in sel_sums:
            queries = []
            for fact in facts:
                query = AggQuery.from_fact(
                    self.table, self.all_preds, 
                    self.cmp_pred, self.agg_cols, fact)
                queries.append(query)
            
            if not [q for q in queries if not self.cache.can_answer(q)]:
                text, _ = self.s_gen.generate(facts)
                reward = self.s_eval.evaluate(text)
                text_to_reward[text] = reward
        
        return text_to_reward, stats

    def _learn_on_sample(self):
        """ Learn good summaries for a data sample.
        
        Returns:
            summaries sorted by estimated quality (descending), statistics
        """
        sample_case = self.test_case.copy()
        table_sample = create_sample(
            self.connection, self.table, 
            self.sample_ratio)
        sample_case['table'] = table_sample
        cmp_pred = sample_case['cmp_pred']
        del sample_case['cmp_pred']
        sample_case['cmp_preds'] = [cmp_pred]
        env = nminer.algs.rl.PickingEnv(
            self.connection, **sample_case, 
            all_preds=self.all_preds, 
            c_type=self.c_type, cluster=True)
        model = A2C(
            'MlpPolicy', env, verbose=True, 
            gamma=1.0, normalize_advantage=True)
        model.learn(total_timesteps=200)
        
        return self._ranked_sums(env), env.statistics()
    
    def _ranked_sums(self, env):
        """ Extracts summaries from environment and ranks them.
        
        Args:
            env: reinforcement learning environment
        
        Returns:
            list of summaries in decreasing order of preference
        """
        prop_sums = env.props_to_rewards.keys()
        sum_eval = []
        for prop_sum in prop_sums:
            reward = env.props_to_rewards[prop_sum]
            conf = env.props_to_conf[prop_sum]
            if reward is not None and conf is not None:
                c_qual = reward * conf
                fact_sum = [nminer.text.fact.Fact.from_props(p) for p in prop_sum]
                sum_eval += [(fact_sum, c_qual)]
        
        s_sums = sorted(sum_eval, key=lambda s_e: s_e[1], reverse=True)
        return [s[0] for s in s_sums]
            
    def _similarity(self, g_query_1, g_query_2):
        """ Calculates similarity (from 0 to 1) between two queries.
        
        Args:
            g_query_1: first group-by query
            g_query_2: second group-by query
        
        Returns:
            similarity between 0 and 1
        """
        if g_query_1.dims == g_query_2:
            preds_1 = g_query_1.preds.copy()
            preds_2 = g_query_2.preds.copy()
            c_union = len(preds_1.union(preds_2))
            c_inter = len(preds_1.intersect(preds_2))
            return c_inter / c_union
        else:
            return 0
                
    def run_sampling(self):
        """ Run sampling algorithm. 
        
        Returns:
            summaries with quality, performance statistics
        """
        start_s = time.time()
        s_sums, p_stats = self._learn_on_sample()
        fs_to_r, m_stats = self._full_data_summaries(s_sums)
        total_s = time.time() - start_s
        
        logging.debug(f'Optimization took {total_s} seconds')
        p_stats['time'] = total_s
        p_stats.update(m_stats)
        
        return fs_to_r, p_stats