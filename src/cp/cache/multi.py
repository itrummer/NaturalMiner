'''
Created on Aug 18, 2021

@author: immanueltrummer
'''
from cp.cache.common import AggCache
from cp.sql.pred import is_pred, pred_sql
from cp.sql.query import AggQuery
from cp.text.fact import Fact
import logging

class MultiItemCache(AggCache):
    """ Proactively caches results for multiple items. """
    
    def __init__(self, connection, table, cmp_col, 
                 all_preds, agg_cols, sum_tmp):
        """ Proactively caches results for summary template for all items. 
        
        Args:
            connection: connection to database
            table: execute queries on this table
            cmp_col: each item associated with one value here
            all_preds: all predicates on dimension columns
            agg_cols: aggregation columns
            sum_tmp: list of fact properties
        """
        self.q_to_r = {}
        with connection.cursor() as cursor:
            for props in sum_tmp:
                q_parts = [f'select {cmp_col}']
                fact = Fact.from_props(props)
                agg_idx = fact.get_agg()
                agg = agg_cols[agg_idx]
                q_parts.append(f'avg({agg})/(select avg({agg}) from {table})')
                
                q_parts.append('where')
                w_parts = []
                for p_idx in fact.get_preds():
                    pred = all_preds[p_idx]
                    if is_pred(pred):
                        col, val = pred
                        w_parts.append(pred_sql(col, val))

                w_clause = ' and '.join(w_parts)
                q_parts.append(w_clause)
                q_parts.append(f'group by {cmp_col}')
                sql = ' '.join(q_parts)
                
                cursor.execute(sql)
                rows = cursor.fetchall()
                for row in rows:
                    cmp_val = row[0]
                    rel_avg = row[1]
                    cmp_pred = (cmp_col, cmp_val)
                    agg_q = AggQuery.from_fact(
                        table, all_preds, cmp_pred, 
                        agg_cols, fact)
                    self.q_to_r[agg_q] = rel_avg
            
        logging.debug(f'Multi-item cache content: {self.q_to_r}')
    
    def can_answer(self, query):
        return True if query in self.q_to_r else False
    
    def get_result(self, query):
        return self.q_to_r[query], -1
    
    def statistics(self):
        return {'cache_hits':-1, 'cache_misses':-1}
    
    def update(self):
        pass