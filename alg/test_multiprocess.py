from multiprocessing import Process
import sys
import json

import test
import test_baseline
import test_ablation
import test_dynasept
import train_test_meta

processes = []

if len(sys.argv) >= 2:
    config_filename = sys.argv[1]
else:
    config_filename = "config.json"

with open(config_filename, 'r') as f:
    config = json.load(f)

if len(sys.argv) >= 3:
    N_seeds = int(sys.argv[2])
else:
    config_filename = "config.json"
    N_seeds = config['main']['N_seeds']

if len(sys.argv) >= 4:
    config_suffix = sys.argv[3]
    dir_name_base = config['main']['dir_name'] + "_{}".format(sys.argv[3])
else:
    config_suffix = None
    dir_name_base = config['main']['dir_name']

if len(sys.argv) >= 5:
    dir_idx_start = int(sys.arg[4])
else:
    dir_idx_start = config['main']['dir_idx_start']

config['main']['phase'] = 'test'
seed_base = config['main']['seed']

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
    test_function = test_dynasept.test_function
elif alg_name == 'maml':
    test_function = train_test_meta.train_test_function


for idx_run in range(N_seeds):
    config_copy = config.copy()
    config_copy['main']['seed'] = seed_base + idx_run + (dir_idx_start-1)
    config_copy['main']['dir_name'] = dir_name_base + '_{:1d}'.format(dir_idx_start + idx_run)

    p = Process(target=test_function, args=(config_copy, config_suffix))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
