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
        'cmp_col':'laptop_name',
        'predicates':[
            "laptop_name='Swift 3 SF314-54G-87HB'", 
            "laptop_name='XPS 15 9570'",
            "laptop_name='Swift 1 SF114-32-C4GB'",
            "laptop_name='MacBook Pro (Retina + Touch Bar)'",
            "laptop_name='Pavilion 13-an0001nx'"
            ]
    },
    {
        'general':{
            'table': 'sods19',
            'dim_cols':['satisfaction', 'dependents', 'optimism', 'codereviews', 'unittests', 
                        'languagesused', 'country', 'coarsesalary', 'orgsize', 'jobtype',
                        'experience', 'devrole', 'gender'],
            'agg_cols':['salary'],
            'nr_facts':1, 'nr_preds':2, 'degree':5, 'max_steps':20,
            'preamble':'Among all developers',
            'dims_tmp':['who are <V> with their job', 
                        'who answered "<V>" when asked if they care for dependents',
                        'who answered "<V>" when asked if they feel optimistic',
                        'who answered "<V>" when asked if they do code reviews',
                        'who answered "<V>" when asked if they do unit tests',
                        'who used <V> as programming languages',
                        'from <V>', 'with a salary <V>', 'working for a <V> company',
                        'working <V>', 'who are <V>', 'who work as <V>', 'who are <V>'],
            'aggs_txt':[', the salary']
            },
        'cmp_col':'languagesused',
        'predicates':[
            "languagesused like '%Bash%'",
            "languagesused like '%HTML%'",
            "languagesused like '%Python%'",
            "languagesused like '%C++%'",
            "languagesused like 'C#'"
            ]
    },
    {
        'general':{
            'table':'flights',
            'dim_cols':['flight_date', 'flight_nr', 'start_airport',
                        'destination_airport', 'planned_departure',
                        'actual_departure', 'wheels_off_time',
                        'wheels_on_time', 'planned_arrival_time',
                        'actual_arrival_time', 'cancelled',
                        'cancellation_code', 'diverted',
                        'planned_flight_time', 
                        'distance'],
            'agg_cols':['departure_delay', 'taxi_out_time', 'taxi_in_time',
                        'arrival_delay', 'actual_time', 'air_time',
                        'carrier_delay', 'weather_delay', 'system_delay',
                        'security_delay', 'late_aircraft_delay'],
            'nr_facts':1, 'nr_preds':2, 'degree':5, 'max_steps':20,
            'preamble':'Among all flights',
            'dims_tmp':['on <V>', 'with number <V>', 'from <V>', 
                        'to <V>', 'scheduled for departure at <V>', 
                        'that departed at <V>', 'that took off at <V>',
                        'that landed at <V>', 'scheduled to arrive at <V>',
                        'that arrived at <V>', 'that were <V>',
                        'with cancellation code <V>', 'that were <V>',
                        'with a planned flight time of <V> minutes',
                        'with a flight distance of <V> miles'
                        ],
            'aggs_txt':[', the departure delay', ', the taxi time at departure',
                        ', the taxi time at the destination', 
                        ', the arrival delay', ', the actual flight time',
                        ', the air time', ', the carrier delay',
                        ', weather delay', ', airport operations delay',
                        ', security delay', ', delay to to late aircrafts']
            },
        'cmp_col':'airline',
        'predicates':[
            "airline='NK'", "airline='WN'", "airline='DL'", 
            "airline='AA'", "airline='OH'" 
            ]
    },
    {
        'general':{
            'table':'liquor', 
            'dim_cols':['city', 'countyname', 'categoryname', 
                        'storename', 'vendorname'], 
            'agg_cols':['bottlessold', 'salevalue', 'volumesold'], 
            'nr_facts':1, 'nr_preds':2, 'degree':5, 'max_steps':20, 
            'preamble':'Among all liquors', 
            'dims_tmp':['sold in <V>', 'sold in <V>', 'of type <V>',
                        'sold at <V>', 'produced by <V>'],
            'aggs_txt':[', the number of bottles per sale', 
                        ', the dollar value per sale', ', the volume per sale']
            },
        'cmp_col':'itemname',
        'predicates':[
            "itemname='Johnnie Walker Black'",
            "itemname='Crown Royal Regal Apple Mini'",
            "itemname='Jim Beam Vanilla'",
            "itemname='Smirnoff Vodka 80 Prf'",
            "itemname='Jim Beam'"
            ]
    }
]

def generate_testcases():
    """ Generates test cases for cherry-picking. 
    
    Returns:
        list of batches of test cases on same data
    """
    test_batches = []
    for s in scenarios:
        general = s['general']
        test_cases = []
        test_batches.append(test_cases)
        for cmp_pred in s['predicates']:
            test_case = general.copy()
            test_case['cmp_pred'] = cmp_pred
            test_cases.append(test_case)
        
    return test_batches