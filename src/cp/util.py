'''
Created on Jun 16, 2021

@author: immanueltrummer
'''
def is_pred(pred):
    """ Checks if tuple represents predicate. 
    
    Args:
        pred: column and associated value
        
    Returns:
        true iff predicate represents condition
    """
    return not pred[0].startswith("'")