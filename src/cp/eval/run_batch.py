'''
Created on Aug 15, 2021

@author: immanueltrummer
'''
import argparse
import cp.algs.batch
import cp.algs.sample
import cp.sql.pred
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import statistics
import time
import cp.algs.batch
from cp.algs.batch import IterativeClusters


def log_ic_results(nr_facts, nr_preds, method, avg_s, ic, out_file):
    """ Logs results to result file.
    
    Args:
        nr_facts: number of facts per summary
        nr_preds: number of predicates per fact
        method: string ID of method used
        avg_s: average seconds per item
        ic: iterative cluster processor
        out_file: write output to this file
    """
    for c_id, (_, b_eval) in ic.id_to_be.items():
        for pred, (d_sum, reward) in b_eval.items():
            out_file.write(
                f'{nr_facts},{nr_preds},{method},{c_id},' +\
                f'{avg_s},"{pred}","{d_sum}",{reward}\n')


def log_si_results(nr_facts, nr_preds, method, iteration, avg_s, si, out_file):
    """ Log results of sub-modular, iterative algorithm.
    
    Args:
        nr_facts: number of facts
        nr_preds: number of predicates
        method: string ID of method used
        avg_s: average seconds per item
        si: sub-modular iterative optimizer
        out_file: handle to output file
    """
    for cmp_pred, (d_sum, quality) in si.best_sums.items():
        out_file.write(
            f'{nr_facts},{nr_preds},{method},{iteration},' +\
            f'{avg_s},"{cmp_pred}","{d_sum}",{quality}\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str, help='Path to JSON input')
    parser.add_argument('db', type=str, help='Name of the database')
    parser.add_argument('user', type=str, help='Name of database user')
    parser.add_argument('out_path', type=str, help='Path to output files')
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
                out_file.write(
                    'nrfacts,nrpreds,approach,iteration,' +\
                    'itemtime,pred,text,reward\n')

                for nr_facts, nr_preds in [
                        (3, 1), (3, 2), (3, 3),
                        (2, 1), (2, 2), (2, 3),
                        (1, 1), (1, 2), (1, 3),]:
                    batch['general']['nr_facts'] = nr_facts
                    batch['general']['nr_preds'] = nr_preds
                    
                    # batch_start_s = time.time()
                    # for cmp_pred in batch['predicates']:
                        # test_case = batch['general'].copy()
                        # test_case['cmp_pred'] = cmp_pred
                        #
                        # start_s = time.time()
                        # sampler = cp.algs.sample.Sampler(
                            # connection, test_case, all_preds, 
                            # 0.01, 5, 'proactive')
                        # text_to_reward, _ = sampler.run_sampling()
                        # total_s = time.time() - start_s
                        #
                        # reward = max(text_to_reward.values())
                        # text = max(text_to_reward.keys(), 
                            # key=lambda k:text_to_reward[k])
                        # out_file.write(
                            # f'{nr_facts},{nr_preds},sample,0,{total_s},' +\
                            # f'"{cmp_pred}","{text}",{reward}\n')
                            #
                        # if time.time() - batch_start_s > 14400:
                            # break
    
                    nr_items = len(batch['predicates'])
                    si = cp.algs.batch.SubModularIterative(
                        connection, batch, all_preds)
                    
                    for i in range(6):
                        start_s = time.time()
                        logging.info(f'Starting batch iteration {i}')
                        si.iterate()
                        q_values = [v[1] for v in si.best_sums.values()]
                        logging.info(f'Average quality: {statistics.mean(q_values)}')
                        total_s = time.time() - start_s
                        avg_s = total_s / nr_items
                        log_si_results(
                            nr_facts, nr_preds, 'si', 
                            i, avg_s, si, out_file)