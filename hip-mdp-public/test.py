"""Test script for method in Killian et al. 2016."""

from __future__ import print_function
import pickle, os, json, time, random
import numpy as np
import random
import tensorflow as tf

from BayesianNeuralNetwork import *
from HiPMDP import HiPMDP
from HiPMDP_test import HiPMDP_test

from ExperienceReplay import ExperienceReplay


def test_function(config, config_suffix=None):
    
    config_main = config['main']
    seed = config_main['seed']
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    
    domain = config_main['domain'] # grid, acrobot, hiv
    if domain == '2D':
        domain_tk = 'grid' # they used 'grid' instead of '2D'
    else:
        domain_tk = domain
    
    if config_suffix is None:
        config_filename = "config_{}.json".format(domain)
    else:
        config_filename = "config_{}{}.json".format(domain, config_suffix)
    with open('../alg/{}'.format(config_filename)) as f:
        config_domain = json.load(f)
    test_steps = config_domain['test_steps']
    N_test_instances = config_domain['N_test_instances']
    dir_name = '../results/%s' % config_main['dir_name']
    config_bnn = config[domain]
    num_training_instances = config_bnn['num_batch_instances']
    
    if domain == '2D':
        preset_hidden_params = [{'latent_code':1},{'latent_code':2}]
    elif domain == 'acrobot' or domain == 'hiv':
        with open('preset_parameters/%s'%config_domain['params_filename'],'r') as f:
            preset_parameters = pickle.load(f)
        if domain == 'acrobot':
            preset_hidden_params = preset_parameters[ : num_training_instances + 2*N_test_instances]
        elif domain == 'hiv':
            # HIV currently only has one instance for validation and test
            preset_hidden_params = preset_parameters[ : num_training_instances + N_test_instances]
    
    with open('{}/{}_network_weights'.format(dir_name, domain), 'r') as f:
        network_weights = pickle.load(f)
    
    if config_main['phase'] == 'validation':
        list_acrobot = [8,9,10,11,12]
        list_hiv = [5]
    elif config_main['phase'] == 'test':
        list_acrobot = [13,14,15,16,17]
        list_hiv = [5]
    
    run_type = 'full'
    create_exp_batch = False
    bnn_hidden_layer_size = config_bnn['bnn_hidden_layer_size'] # 25, 32, 32
    bnn_num_hidden_layers = config_bnn['bnn_num_hidden_layers'] # 3, 2, 2
    grid_beta = 0.1
    
    cumulative_reward = np.zeros((test_steps, N_test_instances))
    idx_instance = 0
    list_times = []
    for idx_instance in range(N_test_instances):
    
        if domain == '2D':
            test_inst = random.sample([0,1], 1)[0]
        elif domain == 'acrobot':
            test_inst = list_acrobot[idx_instance]
        elif domain == 'hiv':
            test_inst = list_hiv[idx_instance]
    
        t_start = time.time()
        test_hipmdp = HiPMDP_test(domain_tk,preset_hidden_params,
                                  ddqn_learning_rate=0.0005, 
                                  run_type=run_type,
                                  bnn_hidden_layer_size=bnn_hidden_layer_size,
                                  bnn_num_hidden_layers=bnn_num_hidden_layers,
                                  bnn_network_weights=network_weights,
                                  test_inst=test_inst,
                                  create_exp_batch=create_exp_batch,
                                  grid_beta=grid_beta,print_output=False,
                                  config_domain=config_domain)
        
    
        cumulative_reward[:,idx_instance] = test_hipmdp.test_single_episode(idx_instance)
        list_times.append( time.time() - t_start )    
    
    header = 'Step'
    for idx in range(1, N_test_instances+1):
        header += ',R_%d' % idx
    indices = np.arange(1,test_steps+1).reshape(test_steps,1)
    concated = np.concatenate([indices, cumulative_reward], axis=1)
    save_loc = '_'.join(dir_name.split('_')[:-1])
    if not os.path.isdir(save_loc):
        os.makedirs(save_loc)
    run_number = dir_name.split('_')[-1]
    np.savetxt('%s/test_%s.csv'%(save_loc,run_number), concated, delimiter=',', fmt='%.3e', header=header)
    
    with open('%s/test_time_%s.pkl'%(save_loc,run_number), 'wb') as f:
        pickle.dump(list_times, f)

if __name__ == "__main__":

    with open('config.json', 'r') as f:
        config = json.load(f)
    test_function(config)
        
