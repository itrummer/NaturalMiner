'''
Created on Aug 7, 2021

@author: immanueltrummer
'''
from cp.sql.query import AggQuery, GroupQuery
import cp.cache.static
import cp.text.fact
import cp.algs.rl
import cp.sql.cost
import cp.text.sum
import logging
from stable_baselines3 import A2C
import time

class Sampler():
    """ Selects data summaries via sampling. """
    
    def __init__(self, connection, test_case, all_preds):
        """ Initialize sampler, creates summary generator.
        
        Args:
            connection: connection to database
            test_case: description of scenario
            all_preds: all possible predicates
        """
        self.connection = connection
        self.test_case = test_case
        self.all_preds = all_preds
        self.table = test_case['table']
        self.cmp_pred = test_case['cmp_pred']
        self.dim_cols = test_case['dim_cols']
        self.agg_cols = test_case['agg_cols']
        preamble = test_case['preamble']
        dims_tmp = test_case['dims_tmp']
        aggs_txt = test_case['aggs_txt']
        
        self.cache = cp.cache.proactive.ProCache(
            connection, self.table, self.cmp_pred)
        self.q_engine = cp.sql.query.QueryEngine(
            connection, self.table, 
            self.cmp_pred, self.cache)
        self.s_gen = cp.text.sum.SumGenerator(
            all_preds, preamble, self.dim_cols,
            dims_tmp, self.agg_cols, aggs_txt,
            self.q_engine)
        self.s_eval = cp.text.sum.SumEvaluator()

    def learn_on_sample(self):
        """ Learn good summaries for a data sample.
        
        Returns:
            summaries sorted by estimated quality (descending), statistics
        """
        sample_case = self.test_case.copy()
        full_table = self.test_case['table']
        table_sample = f'(select * from {full_table} limit 10000) as S'
        sample_case['table'] = table_sample
    
        cache = cp.cache.static.EmptyCache()    
        env = cp.algs.rl.PickingEnv(
            self.connection, **sample_case, 
            all_preds=self.all_preds,
            cache=cache, proactive=False)
        model = A2C(
            'MlpPolicy', env, verbose=True, 
            gamma=1.0, normalize_advantage=True)
        model.learn(total_timesteps=200)
    
        s_sums = sorted(
            env.props_to_rewards.items(),
            key=lambda s: s[1], reversed=True)
        return [
            [cp.text.fact.Fact.from_props(p) for p in s[0]]
            for s in s_sums], env.statistics()
    
    def cost(self, g_queries):
        """ Calculates cost estimate for a set of group-by queries.
        
        Args:
            g_queries: group-by queries
        
        Returns:
            accumulated cost for set of group-by queries
        """
        total_cost = 0
        for g_q in g_queries:
            sql = g_q.sql()
            _, cost = cp.sql.cost.estimates(self.connection, sql)
            total_cost += cost
        
        return total_cost
    
    def similarity(self, g_query_1, g_query_2):
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
    
    def fill_cache(self, sam_sums):
        """ Fill cache with query results needed to generate summaries.
        
        Args:
            sam_sums: summaries (facts) selected via sampling
        """
        # collect queries associated with facts
        queries = []
        for fact in sam_sums[0]:
            a_q = AggQuery.from_fact(
                self.table, self.all_preds,
                self.cmp_pred, self.agg_cols, fact)
            queries += [a_q]
        
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
        min_cost = self.cost(g_merged)
        logging.debug(f'Cost before clustering: {min_cost}')
        
        improved = True
        while improved:
            sims = []
            for g1 in g_merged:
                for g2 in g_merged:
                    if not (g1 == g2):
                        sim = self.similarity(g1, g2)
                        sims += [(g1, g2, sim)]
            g1, g2, _ = min(sims, key=lambda s:s[2])
            
            gm = GroupQuery(self.table, g1.dims, self.cmp_pred)
            gm.preds.update(g1.preds)
            gm.preds.update(g2.preds)
            gm.aggs.update(g1.aggs)
            gm.aggs.update(g2.aggs)
            g_merged.remove(g1)
            g_merged.remove(g2)
            g_merged.add(gm)
            
            cost = self.cost(g_merged)
            improved = False
            if cost < min_cost:
                min_cost = cost
                min_g_qs = g_merged.copy()
                improved = True
        
        logging.debug(f'Cost after clustering: {min_cost} with {min_g_qs}')
        for g_q in min_g_qs:
            self.cache.cache(g_q)
    
    def full_data_summaries(self, sam_sums):
        """ Generates summaries by considering the full data set.
        
        Args:
            sam_sums: summaries selected via sample
        
        Returns:
            a subset of summaries referring to full data set
        """
        text_to_reward = {}
    
        if sam_sums:
            self.fill_cache(sam_sums)
        
        nr_sums = min(len(sam_sums), 5)
        for facts in sam_sums[0:nr_sums]:
            
            queries = []
            for fact in facts:
                query = AggQuery.from_fact(
                    self.table, self.all_preds, 
                    self.cmp_pred, self.agg_cols, fact)
                queries.append(query)
            
            if not [q for q in queries if not self.cache.can_answer(q)]:
                text = self.s_gen.generate(facts)
                reward = self.s_eval.evaluate(text)
                text_to_reward[text] = reward
        
        return text_to_reward
    
    def run_sampling(self):
        """ Run sampling algorithm. 
        
        Returns:
            summaries with quality, performance statistics
        """
        start_s = time.time()
        s_sums, p_stats = self.learn_on_sample()
        fs_to_r = self.full_data_summaries(s_sums)
        total_s = time.time() - start_s
        
        logging.debug(f'Optimization took {total_s} seconds')
        p_stats['time'] = total_s
        p_stats.update(self.q_engine.statistics())
        p_stats.update(self.s_gen.statistics())
        p_stats.update(self.s_eval.statistics())
        
        return fs_to_r, p_stats