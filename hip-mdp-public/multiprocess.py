"""Runs multiple independent training or test runs with different seeds."""

from multiprocessing import Process
import json

import train
import test

processes = []

import sys
if len(sys.argv) > 1:
    config_filename = sys.argv[1]
else:
    config_filename = "config.json"
with open(config_filename, 'r') as f:
    config = json.load(f)

if len(sys.argv) >= 5:
    N_seeds = int(sys.argv[2])
    config_suffix = sys.argv[3]
    dir_name_base = config['main']['dir_name'] + "_{}".format(sys.argv[3])
    phase = sys.argv[4]
else:
    config_suffix = None
    N_seeds = config['main']['N_seeds']
    dir_name_base = config['main']['dir_name']
    phase = config['main']['phase']

if len(sys.argv) >= 6:
    dir_idx_start = int(sys.argv[5])
else:
    dir_idx_start = config['main']['dir_idx_start']

seed_base = config['main']['seed']

if phase == 'train':
    function = train.train_function
else:
    function = test.test_function

for idx_run in range(N_seeds):
    config_copy = config.copy()
    config_copy['main']['seed'] = seed_base + idx_run + (dir_idx_start-1)
    config_copy['main']['dir_name'] = dir_name_base + '_{:1d}'.format(dir_idx_start + idx_run)

    p = Process(target=function, args=(config_copy, config_suffix))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
