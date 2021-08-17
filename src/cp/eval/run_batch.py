'''
Created on Aug 15, 2021

@author: immanueltrummer
'''
import argparse
import cp.algs.batch
import cp.sql.pred
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import time


def log_results(prefix, total_s, result, out_file):
    """ Logs results to result file.
    
    Args:
        prefix: start each line with this prefix
        total_s: total seconds for entire batch
        result: result of processing batch
        out_file: write output to this file
    """
    nr_items = len(result)
    avg_s = total_s/nr_items
    print(f'Total time: {total_s}; ' \
          f'Nr. items: {nr_items}; ' \
          f'Avg. time: {avg_s}')
    
    for pred, (d_sum, reward) in result.items():
        out_file.write(f'{prefix},{avg_s},{pred},{d_sum},{reward}\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str, help='Path to JSON input')
    parser.add_argument('db', type=str, help='Name of the database')
    parser.add_argument('user', type=str, help='Name of database user')
    parser.add_argument('out_path', type=str, help='Path to output file')
    parser.add_argument('log_level', type=str, help='Specify log level')
    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    logging.basicConfig(level=log_level, filemode='w')
    
    with open(args.in_path) as in_file:
        batch = json.load(in_file)
        with open(args.out_path, 'w') as out_file:
            with psycopg2.connect(
                database=args.db, user=args.user, 
                cursor_factory=RealDictCursor) as connection:
                connection.autocommit = True
                all_preds = cp.sql.pred.all_preds(
                    connection, batch['general']['table'], 
                    batch['general']['dim_cols'], 'true')

            # start_s = time.time()
            # solution = cp.algs.batch.simple_batch(
                # connection, batch, all_preds)
            # result = cp.algs.batch.eval_solution(
                # connection, batch, all_preds, solution)
            # total_s = time.time() - start_s
            # log_results('simple', total_s, result, out_file)
            
            start_s = time.time()
            bp = cp.algs.batch.BatchProcessor(connection, batch, all_preds)
            solution = bp.summarize()
            result = cp.algs.batch.eval_solution(
                connection, batch, all_preds, solution)
            total_s = time.time() - start_s
            log_results('clusters', total_s, result, out_file)