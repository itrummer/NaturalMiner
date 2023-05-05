'''
Created on Aug 8, 2021

@author: immanueltrummer
'''
def cardinality(connection, table):
    """ Returns cardinality estimate. 
    
    Args:
        connection: connection to database
        table: estimate rows in this table
    
    Returns:
        cardinality estimate of given table
    """
    sql = f'select * from {table}'
    cardinality, _ = estimates(connection, sql)
    return cardinality

def estimates(connection, sql):
    """ Generates cost and cardinality estimates.
    
    Args:
        connection: connection to database
        sql: an sql command
    
    Returns:
        tuple with output cardinality and cost estimates
    """
    explain_sql = f'explain (format json) {sql}'
    with connection.cursor() as cursor:
        cursor.execute(explain_sql)
        result = cursor.fetchall()
        rows = result[0]['QUERY PLAN'][0]['Plan']['Plan Rows']
        cost = result[0]['QUERY PLAN'][0]['Plan']['Total Cost']
        return rows, cost