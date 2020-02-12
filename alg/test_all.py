from multiprocessing import Process
import json

import test
import test_baseline
import test_ablation
import test_continuous
import train_test_meta
import sys
import os

for config_filename in os.listdir("configs"):
    if 'bnn' in config_filename:
        continue
    processes = []
    print(config_filename)
    with open('configs/'+config_filename) as f:
        config = json.load(f)

    config['main']['phase'] = 'test'
    if len(sys.argv) > 1:
        N_seeds = int(sys.argv[1])
    else:
        N_seeds = config['main']['N_seeds']
    seed_base = config['main']['seed']
    dir_name_base = config['main']['dir_name']
    dir_idx_start = config['main']['dir_idx_start']

    alg_name = config['main']['alg_name']
    if alg_name == 'sept':
        test_function = test.test_function
    elif alg_name == 'avg' or alg_name == 'oracle':
        test_function = test_baseline.test_function
    elif alg_name == 'epopt':
        test_function = test_baseline.test_function
    elif alg_name == 'noexp':
        test_function = test_ablation.test_function
    elif alg_name == 'dynasept':
        test_function = test_continuous.test_function
    elif alg_name == 'maml':
        test_function = train_test_meta.train_test_function

    for idx_run in range(N_seeds):
        config_copy = config.copy()
        config_copy['main']['seed'] = seed_base + idx_run
        config_copy['main']['dir_name'] = dir_name_base + '_{:1d}'.format(dir_idx_start + idx_run)

        p = Process(target=test_function, args=(config_copy,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
