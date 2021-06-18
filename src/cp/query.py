'''
Created on Jun 5, 2021

@author: immanueltrummer
'''
from collections import Counter
from cp.util import is_pred
import psycopg2
import time

class AggQuery():
    """ Represents a simple aggregation query. """
    
    def __init__(self, table, eq_preds, cmp_pred, agg_col):
        """ Initializes aggregation query. 
        
        Args:
            table: table the query refers to
            eq_preds: list of equality predicates
            cmp_pred: predicate for comparison
            agg_col: aggregation column
        """
        self.table = table
        self.eq_preds = eq_preds
        self.cmp_pred = cmp_pred
        self.agg_col = agg_col
        
    def pred_cols(self):
        """ Returns columns of query predicates. 
        
        Returns:
            frozen set with predicate columns
        """
        return frozenset([p[0] for p in self.eq_preds if is_pred(p)])
        
    def template(self):
        """ Calculate query template signature for cache lookups.
        
        Returns:
            query template signature as tuple
        """
        cols = self.pred_cols()
        return self.table, cols, self.cmp_pred, self.agg_col


class AggCache():
    """ Cache of aggregate values. """

    def __init__(self, connection):
        """ Initializes query cache.
        
        Args:
            connection: database connection
        """
        self.connection = connection
        self.prefix = 'pcache'
        self.max_cached = 100
        self.t_to_slot = {}
        self.query_log = []
        self.clear_cache()

    def clear_cache(self):
        """ Clears all cached relations. """
        self.next_slot = 0
        for i in range(self.max_cached):
            with self.connection.cursor() as cursor:
                cache_tbl = self.prefix + str(i)
                sql = f'drop table if exists {cache_tbl}'
                cursor.execute(sql)

    def estimate_cardinality(self, template):
        pass

    def get_result(self, query):
        """ Retrieve result from cache. 
        
        Args:
            query: retrieve result for this query
            
        Returns:
            query result (a relative average)
        """
        template = query.template()
        slot = self.t_to_slot[template]
        table = self.prefix + str(slot)
        
        p_parts = [f"{c}='{v}'" for c, v in query.eq_preds]
        w_clause = ' and '.join(p_parts)
        sql = f'select case when cmp_c = 0 or s = 0 then NULL '\
            f'else (cmp_s/cmp_c)/(s/c) end as rel_avg ' \
            f'from {table} where {w_clause}'
        
        with self.connection.cursor() as cursor:
            start_s = time.time()
            cursor.execute(sql)
            print(f'Get time: {time.time() - start_s} seconds')
            if cursor.rowcount == 0:
                return None
            else:
                return cursor.fetchone()[0]

    def is_cached(self, query):
        """ Checks if result of query is cached. 
        
        Args:
            query: check for result of this query
            
        Returns:
            Boolean flag, true iff result is cached
        """
        self.query_log.append(query)
        template = query.template()
        return True if template in self.t_to_slot else False

    def put_results(self, template):
        """ Generates and caches results for query template.
        
        Args:
            template: query template for which to store results
        """
        table, pred_cols, cmp_pred, agg_col = template
        cache_tbl = self.prefix + str(self.next_slot)
        q_parts = [f'create unlogged table {cache_tbl} as (']
        
        s_parts = [f'select sum({agg_col}) as s, count(*) as c']
        s_parts += [f'sum(case when {cmp_pred} then {agg_col} else 0 end) as cmp_s']
        s_parts += [f'sum(case when {cmp_pred} then 1 else 0 end) as cmp_c']
        s_parts += list(pred_cols)
        
        q_parts += [', '.join(s_parts)]
        q_parts += [f' from {table}']
        if pred_cols:
            q_parts += [' group by ' + ', '.join(pred_cols)]
        q_parts += [')']
        
        sql = ' '.join(q_parts)
        print(f'About to fill cache with SQL "{sql}"')
        with self.connection.cursor() as cursor:
            start_s = time.time()
            cursor.execute(sql)
            print(f'Put time: {time.time() - start_s} seconds')
        
        self.t_to_slot[template] = self.next_slot
        self.next_slot += 1

    def _merge_templates(self, t_1, t_2):
        """ Merge templates into a new template.
        
        Args:
            t_1: first template to merge
            t_2: second template to merge
            
        Returns:
            template combining columns if compatible, otherwise None
        """
        table_1, col_1, pred_1, agg_1 = t_1
        table_2, col_2, pred_2, agg_2 = t_2
        if table_1 == table_2 and pred_1 == pred_2 and agg_1 == agg_2:
            merged_cols = set()
            merged_cols.update(col_1)
            merged_cols.update(col_2)
            return table_1, merged_cols, pred_1, agg_1
        else:
            return None

    def _candidate_views(self):
        """ Selects candidate templates for which to generate results. 
        
        Returns:
            set of query templates representing candidates
        """
        t_counts = Counter()
        for q in self.query_log:            
            t = q.template()
            t_counts.update(t)
            
        candidates = set(t_counts.most_common(10))
        for _ in range(3):
            expanded = set()
            for t_1 in candidates:
                for t_2 in candidates:
                    t_m = self._merge_templates(t_1, t_2)
                    expanded.add(t_m)
            candidates.update(expanded)
        
        return candidates

    def _views_to_add(self, candidates):
        """ Select most interesting views to add.
        
        Args:
            candidates: select among those views
            
        Returns:
            best views to add
        """
        pass
    
    def _views_to_delete(self, to_add):
        """ Select least interesting views to delete. 
        
        Args:
            to_add: views to add
            
        Returns:
            best views to delete
        """
        pass
    
    def _order_updates(self, to_add, to_delete):
        """ Order additions and deletions to minimize cost. 
        
        Args:
            to_add: views to be added
            to_delete: views to be deleted
            
        Returns:
            list of plan steps (action-template pairs)
        """
        pass

    def _execute_plan(self, plan):
        """ Execute a plan to update cache.
        
        Args:
            plan: list of plan steps (action-template pairs)
        """
        pass

    def update(self):
        """ Updates cache content for maximum efficiency. """
        candidates = self._candidate_views()
        v_add = self._views_to_add(candidates)
        v_del = self._views_to_delete(v_add)
        plan = self._order_updates(v_add, v_del)
        self._execute_plan(plan)
        self.query_log.clear()

class QueryEngine():
    """ Processes queries distinguishing entities from others. """
    
    def __init__(self, dbname, dbuser, dbpwd, table, cmp_pred):
        """ Initialize query engine for specific connection.
        
        Args:
            dbname: name of database to query
            dbuser: database user name
            dbpwd: database login password
            table: queries refer to this table
            cmp_pred: use for comparisons
        """
        self.dbname = dbname
        self.dbuser = dbuser
        self.dbpwd = dbpwd
        self.table = table
        self.cmp_pred = cmp_pred
        self.connection = psycopg2.connect(database=dbname, user=dbuser, password=dbpwd)
        self.connection.autocommit = True
        self.q_cache = AggCache(self.connection)
        
    def avg(self, eq_preds, pred, agg_col):
        """ Calculate average over aggregation column in scope. 
        
        Args:
            eq_preds: equality predicates as column-value pairs
            pred: SQL string representing predicate
            agg_col: calculate average for this column
            
        Returns:
            Average over aggregation column for satisfying rows
        """
        q_parts = [f'select avg({agg_col}) from {self.table} where TRUE'] 
        q_parts += [f"{c}='{v}'" for c, v in eq_preds]
        q_parts += [pred]
        query = ' AND '.join(q_parts)
        
        with psycopg2.connect(dbname=self.dbname, 
                              user=self.dbuser, 
                              password=self.dbpwd) as connection:
            with connection.cursor() as cursor:
                print(f'About to execute query {query}')
                cursor.execute(query)
                avg = cursor.fetchone()[0]
            
        return avg
    
    def rel_avg(self, eq_preds, agg_col):
        """ Relative average of focus entity in given data scope. 
        
        Args:
            eq_preds: equality predicates defining scope
            agg_col: consider averages in this column
            
        Returns:
            Ratio of entity to general average
        """
        query = AggQuery(self.table, eq_preds, self.cmp_pred, agg_col)
        if not self.q_cache.is_cached(query):
            template = query.template()
            self.q_cache.put_results(template)
        
        return self.q_cache.get_result(query)
        
        # entity_avg = self.avg(eq_preds, self.cmp_pred, agg_col)
        # general_avg = self.avg(eq_preds, 'true', agg_col)
        #
        # if entity_avg is None:
            # return None
        # else:
            # f_gen_avg = max(0.0001, float(general_avg))
            # return float(entity_avg) / f_gen_avg