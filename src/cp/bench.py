'''
Created on Jul 21, 2021

@author: immanueltrummer
'''
scenarios = [
    {
        'general':{
            'table':'laptops', 
            'dim_cols':['brand', 'processor_type', 'graphics_card', 'disk_space', 'ratings_5max'], 
            'agg_cols':['display_size', 'discount_price', 'old_price'], 
            'nr_facts':1, 'nr_preds':2, 'degree':5, 'max_steps':20,
            'preamble':'Among all laptops', 
            'dims_tmp':['with <V> brand', 'with <V> CPU', 'with <V> graphics card', 
                        'with <V> disk space', 'with <V> stars'],
            'aggs_txt':['its display size', 'its discounted price', 'its old price']
            },
        'instances':{
            'keycol':'laptop_name', 
            'entities':['"Swift 3 SF314-54G-87HB"', '"XPS 15 9570"', '"Swift 1 SF114-32-C4GB"']
            }
    },
    {
        'general':{
            'table':'liquor', 
            'dim_cols':['city', 'countyname', 'categoryname', 'itemname'], 
            'agg_cols':['bottlessold', 'salevalue', 'volumesold'], 
            'nr_facts':1, 'nr_preds':2, 'degree':5, 'max_steps':20, 
            'preamble':'Among all stores', 
            'dims_tmp':['in <V>', 'in <V>', 'considering <V>'],
            'aggs_txt':[', the number of bottles per sale', 
                        ', the dollar value per sale', ', the volume per sale']
            },
        'instances':{
            'keycol':'itemname', 
            'entities':['"Barr Hill Gin"', '"Effen Vodka Cubs Edition"', '"Bacardi 151 Rum"']
            }
    },
    {
        'general':{        
            'table':'vehicles', 
            'dim_cols':['region', 'year', 'manufacturer', 'model', 'condition', 
             'cylinders', 'fuel', 'title_status', 'transmission', 'drive',
             'size', 'type', 'paint_color', 'county', 'state'],
            'agg_cols':['price', 'odometer'], 'nr_facts':1, 'nr_preds':2, 
            'degree':5, 'max_steps':20, 'preamble':'Among all cars',
            'dims_tmp':['in <V>', 'from <V>', 'from <V>', 'for model <V>', 
                      'in <V> condition', 'with <V> cylinders', 'with <V> engine',
                      'with <V>', 'with <V> status', 'with <V> transmission', 
                      'with <V>', 'of size <V>', 'of type <V>', 'in <V>', 
                      'in <V>', 'in <V>'],
            'aggs_txt':[', the price', ', the number of miles traveled']
            },
        'instances':{
            'keycol':'id', 'entities':[7310898806, 7313312004, 7303589805]
            }
    },
    {
        'general':{        
            'table':'melbournedec18', 
            'dim_cols':['host_name', 'neighbourhood', 'neighbourhood_group', 'room_type'],
            'agg_cols':['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                      'calculated_host_listings_count', 'availability_365'],
            'nr_facts':1, 'nr_preds':2, 'degree':5, 'max_steps':20,
            'preamble':'Among all apartments, ', 
            'dims_tmp':['hosted by <V>', 'in <V>', 'in <V>', 'of type <V>'],
            'aggs_txt':[', the price', ', the minimum stay', ', the number of reviews',
                      ', the reviews per month', ', the number of listings by the same host',
                      ', the number of available days over the year']
            },
        'instances':{
            'keycol':'id', 'entities':[17701730, 9601604, 20425757]
            }
    }
]