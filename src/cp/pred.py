'''
Created on Jun 16, 2021

@author: immanueltrummer
'''
def all_preds(connection, table, dim_cols, cmp_pred):
    """ Extracts all possible predicates from database. 
    
    Args:
        connection: connection to database
        table: extract predicates for this table
        dim_cols: list of dimension columns
        cmp_pred: filter rows by this predicate
    
    Returns:
        list of (column, value) pairs representing predicates    
    """
    preds = [("'any'", 'any')]
    for dim in dim_cols:
        print(f'Generating predicates for dimension {dim} ...')
        with connection.cursor() as cursor:
            query = f'select distinct {dim} from {table} ' \
                f'where {cmp_pred} and {dim} is not null'
            cursor.execute(query)
            result = cursor.fetchall()
            preds += [(dim, r[0]) for r in result]
    return preds

def is_pred(pred):
    """ Checks if tuple represents predicate. 
    
    Args:
        pred: column and associated value
        
    Returns:
        true iff predicate represents condition
    """
    return not pred[0].startswith("'")

def pred_sql(col, val):
    """ Generates SQL for equality predicate.
    
    Args:
        col: predicate restricts this column
        val: filter column using this value
        
    Returns:
        predicate as SQL string (escaped)
    """
    esc_val = str(val).replace("'", r"''")
    return f"{col}='{esc_val}'"