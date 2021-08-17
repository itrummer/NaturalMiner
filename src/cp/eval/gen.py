'''
Created on Aug 15, 2021

@author: immanueltrummer
'''
import argparse
import cp.eval.bench
import cp.sql.pred
import json
import psycopg2


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('db', type=str, help='Name of database to access')
    parser.add_argument('user', type=str, help='Database user name')
    args = parser.parse_args()
    
    with psycopg2.connect(database=args.db, user=args.user) as connection:
        for scenario_id, scenario in enumerate(cp.eval.bench.scenarios):
            print(f'Processing scenario {scenario_id}')
            
            cmp_col = scenario['cmp_col']
            with connection.cursor() as cursor:
                table = scenario['general']['table']
                sql = f'select distinct {cmp_col} from {table} ' \
                    f'where {cmp_col} is not null'
                cursor.execute(sql)
                cmp_vals = [v[0] for v in cursor.fetchall()]
                
                if scenario_id == 1:
                    languages = set()
                    for v in cmp_vals:
                        if v is not None:
                            for s in v.split(';'):
                                languages.add(s)
                    preds = [f"{cmp_col} like '%{cp.sql.pred.sql_esc(l)}%'" 
                             for l in languages]
                else:
                    preds = [cp.sql.pred.pred_sql(cmp_col, v) for v in cmp_vals]
                
                scenario['predicates'] = preds
                print(f'Extracted {len(preds)} predicates.')
                json_path = f'bench/batch{scenario_id}.json'
                with open(json_path, 'w') as file:
                    json.dump(scenario, file)