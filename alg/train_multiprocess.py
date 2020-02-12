"""
Note that ablations totvar and negvae must still use train.train_function
not train_ablation
"""

from multiprocessing import Process
from copy import deepcopy
import sys
import json

import train
import train_baseline
import train_ablation
import train_dynasept
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
    N_seeds = config['main']['N_seeds']

if len(sys.argv) >= 4:
    config_suffix = sys.argv[3]
    dir_name_base = config['main']['dir_name'] + "_{}".format(sys.argv[3])
else:
    config_suffix = None
    dir_name_base = config['main']['dir_name']

if len(sys.argv) >= 5:
    dir_idx_start = int(sys.argv[4])
else:
    dir_idx_start = config['main']['dir_idx_start']


# Set command-line args
config['main']['N_seeds'] = N_seeds
config['main']['dir_idx_start'] = dir_idx_start


seed_base = config['main']['seed']

alg_name = config['main']['alg_name']
if alg_name == 'sept':
    train_function = train.train_function
elif alg_name == 'avg' or alg_name == 'oracle':
    train_function = train_baseline.train_function
elif alg_name == 'epopt':
    import train_epopt
    train_function = train_epopt.train_function
elif alg_name == 'noexp':
    train_function = train_ablation.train_function
elif alg_name == 'dynasept':
    train_function = train_dynasept.train_function
elif alg_name == 'maml':
    train_function = train_test_meta.train_test_function


for idx_run in range(N_seeds):
    config_copy = deepcopy(config)
    config_copy['main']['seed'] = seed_base + idx_run + (dir_idx_start-1)
    config_copy['main']['dir_name'] = dir_name_base + '_{:1d}'.format(dir_idx_start + idx_run)

    p = Process(target=train_function, args=(config_copy, config_suffix))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
