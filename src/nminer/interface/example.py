'''
Created on May 5, 2023

@author: immanueltrummer
'''
import nminer.interface.mine

# Name of Postgres database to mine
db_name = 'picker'
# User name for Postgres database
db_user = 'root'
# Password for Postgres database
db_pwd = ''
# Mine data in this table
table ='laptops'
# Table columns (typically categorical) for equality predicates
dim_cols = [
    'brand', 'processor_type', 'graphics_card', 'disk_space', 'ratings_5max']
# Text templates to express equality predicates associated with dimension
# columns. The <V> placeholder will be substituted by the predicate constant.
dims_txt = [
        'with <V> brand', 'with <V> CPU', 'with <V> graphics card', 
        'with <V> disk space', 'with <V> stars ']
# Table columns (typically numerical) for calculating aggregates.
agg_cols = ['display_size', 'discount_price', 'old_price']
# Text templates used to express aggregates on aforementioned columns.
aggs_txt = ['its display size', 'its discounted price', 'its old price']
# An SQL predicate describing the target data: we mine for facts that compare
# rows satisfying that predicate to all rows. Here, target data is associated
# with a specific laptop model and facts compare this to other laptops.
target = "laptop_name='VivoBook S430'"
# All facts start with this text, specifying what we compare to.
preamble = 'Among all laptops'
# Search for facts that comply with a pattern, described in natural language.
# E.g., we can search for arguments for or against buying this laptop.
nl_pattern = "arguments for buying this laptop"

# Mine for facts that are relevant, given the pattern
facts, relevance = nminer.interface.mine.mine(
    db_name, db_user, db_pwd, table,
    preamble, dim_cols, dims_txt, agg_cols, aggs_txt, 
    target, nl_pattern)

# Output mining results
print(f'Mined facts: {facts}')
print(f'Relevance: {relevance}')