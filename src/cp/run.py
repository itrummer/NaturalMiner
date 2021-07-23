'''
Created on Jun 6, 2021

@author: immanueltrummer
'''
import cp.base
import cp.bench
import cp.rl
import psycopg2
import time

from stable_baselines3 import A2C, PPO
from cp.pred import all_preds

def print_details(env):
    """ Prints details about RL results.
    
    Args:
        env: extract info from this environment
    """
    best_summary = env.best_summary()
    print(f'Best summary: "{best_summary}"')
    for best in [True, False]:
        print(f'Top-K summaries (Best: {best})')
        topk_summaries = env.topk_summaries(2, best)
        for s in topk_summaries:
            print(s)
            
    print('Facts generated:')
    for f in env.fact_to_text.values():
        print(f'{f}')


def run_rl(connection, test_case, all_preds):
    """ Benchmarks primary method on test case.
    
    Args:
        connection: connection to database
        test_case: describes test case
        all_preds: ordered predicates
        
    Returns:
        summaries with reward, performance statistics
    """
    start_s = time.time()
    env = cp.rl.PickingEnv(connection, **test_case, all_preds=all_preds)
    model = A2C(
        'MlpPolicy', env, verbose=True, 
        gamma=1.0, normalize_advantage=True)
    model.learn(total_timesteps=30) # 10000
    total_s = time.time() - start_s
    print(f'Optimization took {total_s} seconds')
    
    p_stats = {'time':total_s}
    p_stats.update(env.statistics())
    
    return env.s_eval.text_to_reward, p_stats


def run_random(connection, test_case, all_preds, nr_sums, timeout_s):
    """ Run simple random generation baseline.
    
    Args:
        connection: connection to database
        test_case: summarize for this test case
        all_preds: all available predicates
        nr_sums: generate so many summaries
        timeout_s: sample until this timeout
        
    Returns:
        summaries with quality, performance statistics
    """
    return cp.base.rand_sums(
        nr_sums=nr_sums, timeout_s=timeout_s, 
        connection=connection, all_preds=all_preds,
        **test_case)

def log_line(outfile, b_id, t_id, m_id, sums, p_stats):
    """ Writes one line to log file.
    
    Args:
        outfile: pointer to output file
        b_id: batch ID
        t_id: test case ID
        m_id: method identifier
        sums: maps summaries to rewards
        p_stats: performance statistics
    """
    sorted_sums = sorted(sums.items(), key=lambda s: s[1])
    b_sum = sorted_sums[-1]
    w_sum = sorted_sums[0]
    
    e_time = p_stats['evaluation_time']
    time = p_stats['time']
    hitrate = 0

    outfile.write(f'{b_id}\t{t_id}\t{m_id}\t' \
                  f'{b_sum[0]}\t{b_sum[1]}\t{w_sum[0]}\t{w_sum[1]}\t' \
                  f'{time}\t{e_time}\t{hitrate}\n')
    outfile.flush()


def main():
    db = 'picker'
    user = 'immanueltrummer'
    outpath = 'cpout.tsv'
    
    with open(outpath, 'w') as file:
        file.write('scenario\ttestcase\tapproach\t' \
                   'best\tbquality\tworst\twquality\t'\
                   'time\tetime\thitrate\n')
        with psycopg2.connect(database=db, user=user) as connection:
            connection.autocommit = True
            
            test_batches = cp.bench.generate_testcases()
            for b_id, b in enumerate(test_batches):
                for t_id, t in enumerate(b):
                    
                    print(f'Next up: {b_id}/{t_id}')
                    all_preds = cp.pred.all_preds(
                        connection, t['table'], 
                        t['dim_cols'], t['cmp_pred'])
                    
                    sums, p_stats = run_rl(connection, t, all_preds)
                    log_line(file, b_id, t_id, 'rl', sums, p_stats)
                    timeout_s = p_stats['time']
                    
                    sums, p_stats = run_random(
                        connection, t, all_preds, 1, float('inf'))
                    log_line(file, b_id, t_id, 'rand1', sums, p_stats)
                    
                    sums, p_stats = run_random(
                        connection, t, all_preds, float('inf'), timeout_s)
                    log_line(file, b_id, t_id, 'rand', sums, p_stats)

if __name__ == '__main__':
    main()