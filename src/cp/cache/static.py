'''
Created on Jul 31, 2021

@author: immanueltrummer
'''
from cp.cache.common import AggCache
from cp.sql.pred import is_pred, pred_sql
import logging
import time

class EmptyCache(AggCache):
    """ Dummy cache that remains empty. """
    
    def can_answer(self, _):
        return False
    
    def get_result(self, _):
        raise NotImplementedError
    
    def update(self):
        pass


class CubeCache(AggCache):
    """ Answers all queries from initially created data cube. """
    
    def __init__(self, connection, src_table, dim_cols, 
                 cmp_pred, agg_cols, timeout_s):
        """ Initializes data cube used for all lookups.
        
        Args:
            connection: database connection
            src_table: build cube from this table
            dim_cols: dimension columns of cube
            cmp_pred: predicate for comparisons
            agg_cols: aggregation columns of cube
            timeout_s: timeout in seconds
        """
        self.connection = connection
        self.src_tbl = src_table
        self.dim_cols = dim_cols
        self.cmp_pred = cmp_pred
        self.agg_cols = agg_cols
        self.timeout_s = timeout_s
        self.cube_tbl = 'cpcube'
        
        self._clear()
        self.have_cube = self._create_cube()
    
    def can_answer(self, _):
        return self.have_cube
    
    def get_result(self, query):
        """ Get result for query from cached cube.
        
        Args:
            query: aggregation query
            
        Returns:
            query result
        """
        a = query.agg_col
        q_parts = [f'select case when cmp_c = 0 or s_{a} = 0 then NULL']
        q_parts += [f'else (cmp_s_{a}/cmp_c)/(s_{a}/c) end as rel_avg']
        q_parts += [f'from {self.cube_tbl}']
        
        w_parts = [pred_sql(c, v) for c, v in query.eq_preds if is_pred((c, v))]
        p_cols = query.pred_cols()
        for dim in self.dim_cols:
            if dim not in p_cols:
                w_parts += [f'{dim} is NULL']
        w_clause = ' AND '.join(w_parts)
        
        sql = ' '.join(q_parts) + ' where ' + w_clause
        with self.connection.cursor() as cursor:
            logging.debug(f'Cache lookup via "{sql}"')
            cursor.execute(sql)
            if cursor.rowcount == 0:
                return None
            else:
                return cursor.fetchone()[0]
    
    def update(self):
        pass
    
    def _clear(self):
        """ Clears cache content. """
        with self.connection.cursor() as cursor:
            cursor.execute(f'drop table if exists {self.cube_tbl}')
    
    def _create_cube(self):
        """ Creates cube used for answering all queries.
        
        Returns:
            true iff cube was created until timeout
        """
        if len(self.dim_cols) > 12:
            return False
        
        s_parts = ['select count(*) as c']
        s_parts += [f'sum(case when {self.cmp_pred} then 1 else 0 end) as cmp_c']
        s_parts += self.dim_cols
        
        for agg_col in self.agg_cols:
            s_parts += [f'sum({agg_col}) as s_{agg_col}']
            s_parts += [f'sum(case when {self.cmp_pred} then {agg_col}' \
                        f' else 0 end) as cmp_s_{agg_col}']
            
        sql = f'create unlogged table {self.cube_tbl} as (' + \
            ', '.join(s_parts) + f' from {self.src_tbl} ' + \
            ' group by cube (' + ', '.join(self.dim_cols) + '))'
        logging.debug(f'About to create cube via "{sql}"')

        start_s = time.time()
        success = False
        with self.connection.cursor() as cursor:
            cursor.execute(f"set statement_timeout = '{self.timeout_s}s'")
            try:
                cursor.execute(sql)
                success = True
            finally:
                cursor.execute('set statement_timeout = 0')
        total_s = time.time() - start_s
        logging.debug(f'Created cube in {total_s} seconds; success: {success}')
            
        return success