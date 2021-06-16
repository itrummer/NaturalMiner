'''
Created on Jun 5, 2021

@author: immanueltrummer
'''
import psycopg2
from cp.util import is_pred

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
        self.clear_cache()
        
    def clear_cache(self):
        """ Clears all cached relations. """
        self.next_slot = 0
        for i in range(self.max_cached):
            with self.connection.cursor() as cursor:
                cache_tbl = self.prefix + str(i)
                sql = f'drop table if exists {cache_tbl}'
                cursor.execute(sql)
        
    def is_cached(self, query):
        """ Checks if result of query is cached. 
        
        Args:
            query: check for result of this query
            
        Returns:
            Boolean flag, true iff result is cached
        """
        template = query.template()
        return True if template in self.t_to_slot else False
        
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
        sql = f'select rel_avg from {table} where {w_clause}'
        
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
            if cursor.rowcount == 0:
                return None
            else:
                return cursor.fetchone()[0]
    
    def put_results(self, template):
        """ Generates and caches results for query template.
        
        Args:
            template: query template for which to store results
        """
        table, pred_cols, cmp_pred, agg_col = template
        cache_tbl = self.prefix + str(self.next_slot)
        q_parts = [f'create unlogged table {cache_tbl} as (']
        q_parts += ['with sums as (']
        
        s_parts = [f'select sum({agg_col}) as s, count(*) as c']
        s_parts += [f'sum(case when {cmp_pred} then {agg_col} else 0 end) as cmp_s']
        s_parts += [f'sum(case when {cmp_pred} then 1 else 0 end) as cmp_c']
        s_parts += list(pred_cols)
        
        q_parts += [', '.join(s_parts)]
        q_parts += [f' from {table}']
        if pred_cols:
            q_parts += [' group by ' + ', '.join(pred_cols)]
        q_parts += [') select case when cmp_c = 0 or s = 0 then NULL ']
        q_parts += ['else (cmp_s/cmp_c)/(s/c) end as rel_avg, * from sums)']
        
        sql = ' '.join(q_parts)
        print(f'About to fill cache with SQL "{sql}"')
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
        
        self.t_to_slot[template] = self.next_slot
        self.next_slot += 1


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