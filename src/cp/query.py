'''
Created on Jun 5, 2021

@author: immanueltrummer
'''
class QueryEngine():
    """ Processes queries distinguishing entities from others. """
    
    def __init__(self, connection, table, cmp_pred):
        """ Initialize query engine for specific connection.
        
        Args:
            connection: connection to database
            table: queries refer to this table
            cmp_pred: use for comparisons
        """
        self.connection = connection
        self.table = table
        self.cmp_pred = cmp_pred
    
    def rel_avg(self, preds, agg_col):
        """ Relative average of focus entity in given data scope. 
        
        Args:
            preds: analyze data in scope defined by predicates
            agg_col: consider averages in this column
            
        Returns:
            Ratio of entity to general average
        """
        entity_avg = self.avg(preds + [self.cmp_pred], agg_col)
        general_avg = self.avg(preds, agg_col)
        
        return entity_avg / general_avg
    
    def avg(self, preds, agg_col):
        """ Calculate average over aggregation column in scope. 
        
        Args:
            preds: consider rows satisfying those predicates
            agg_col: calculate average for this column
            
        Returns:
            Average over aggregation column for satisfying rows
        """
        q_parts = [f'select avg({agg_col}) from {self.table} where TRUE'] + preds
        query = ' AND '.join(q_parts)
        
        with self.connection.cursor() as cursor:
            avg = cursor.execute(query).fetchone()[0]
            
        return avg