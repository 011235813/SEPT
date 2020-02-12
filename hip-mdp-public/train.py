"""Trains the method from Killian et al. 2016."""

from __future__ import print_function
import pickle, os, json, time
import numpy as np
import random
import tensorflow as tf

from BayesianNeuralNetwork import *
from HiPMDP import HiPMDP

from ExperienceReplay import ExperienceReplay


def train_function(config, config_suffix=None):

    config_main = config['main']
    seed = config_main['seed']
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    
    domain = config_main['domain'] # 2D, acrobot, hiv
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
    
    config_bnn = config[domain]
        
    dir_name = '../results/%s' % config_main['dir_name']
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    run_type = 'modelfree'
    num_batch_instances = config_bnn['num_batch_instances'] # 2, 8, 5
    if domain == '2D':
        preset_hidden_params = [{'latent_code':1},{'latent_code':2}]
    elif domain == 'acrobot' or domain == 'hiv':
        with open('preset_parameters/%s'%config_domain['params_filename'],'r') as f:
            preset_parameters = pickle.load(f)
        preset_hidden_params = preset_parameters[:num_batch_instances]
    
    ddqn_learning_rate = 0.0005
    episode_count = 500
    bnn_hidden_layer_size = config_bnn['bnn_hidden_layer_size'] # 25, 32, 32
    bnn_num_hidden_layers = config_bnn['bnn_num_hidden_layers'] # 3, 2, 2
    bnn_network_weights = None
    eps_min = 0.15
    test_inst = None
    create_exp_batch = True
    state_diffs = True
    grid_beta = 0.1
    batch_generator_hipmdp = HiPMDP(domain_tk,preset_hidden_params,
                                    ddqn_learning_rate=ddqn_learning_rate,
                                    episode_count=episode_count,
                                    run_type=run_type, eps_min=eps_min,
                                    create_exp_batch=create_exp_batch,
                                    num_batch_instances=num_batch_instances,
                                    grid_beta=grid_beta,
                                    print_output=True, config_domain=config_domain)
    
    t_start = time.time()
    (exp_buffer, networkweights, rewards, avg_rwd_per_ep, full_task_weights,
         sys_param_set, mean_episode_errors, std_episode_errors) = batch_generator_hipmdp.run_experiment()
    
    
    with open('{}/{}_exp_buffer'.format(dir_name, domain),'w') as f:
        pickle.dump(exp_buffer,f)
    # with open('{}/{}_exp_buffer'.format(dir_name, domain),'r') as f:
    #      exp_buffer = pickle.load(f)
    
    # Create numpy array 
    exp_buffer_np = np.vstack(exp_buffer)
    # Collect the instances that each transition came from
    inst_indices = exp_buffer_np[:,4]
    inst_indices = inst_indices.astype(int)
    # Group experiences by instance
    # Create dictionary where keys are instance indexes and values are np.arrays experiences
    exp_dict = {}
    for idx in xrange(batch_generator_hipmdp.instance_count):
        exp_dict[idx] = exp_buffer_np[inst_indices == idx]
    X = np.array([np.hstack([exp_buffer_np[tt,0],exp_buffer_np[tt,1]]) for tt in range(exp_buffer_np.shape[0])])
    y = np.array([exp_buffer_np[tt,3] for tt in range(exp_buffer_np.shape[0])])
    num_dims = config_domain['n_state'] # 2, 4, 6
    num_actions = config_domain['n_action'] # 4, 3, 4
    num_wb = 5
    if state_diffs:
        # subtract previous state
        y -= X[:,:num_dims]
    
    relu = lambda x: np.maximum(x, 0.)
    param_set = {
        'bnn_layer_sizes': [num_dims+num_actions+num_wb]+[bnn_hidden_layer_size]*bnn_num_hidden_layers+[num_dims],
        'weight_count': num_wb,
        'num_state_dims': num_dims,
        'bnn_num_samples': 50,
        'bnn_batch_size': 32,
        'num_strata_samples': 5,
        'bnn_training_epochs': 1,
        'bnn_v_prior': 1.0,
        'bnn_learning_rate': config_bnn['bnn_learning_rate'], # 5e-5, 2.5e-4, 2.5e-4
        'bnn_alpha': config_bnn['bnn_alpha'], # 0.5, 0.5, 0.45
        'wb_num_epochs':1,
        'wb_learning_rate':0.0005
    }
    # Initialize latent weights for each instance
    full_task_weights = np.random.normal(0.,0.1,(batch_generator_hipmdp.instance_count,num_wb))
    # Initialize BNN
    network = BayesianNeuralNetwork(param_set, nonlinearity=relu)
    
    # Compute error before training
    l2_errors = network.get_td_error(np.hstack((X,full_task_weights[inst_indices])), y, location=0.0, scale=1.0, by_dim=False)
    print ("Before training: Mean Error: {}, Std Error: {}".format(np.mean(l2_errors),np.std(l2_errors)))
    np.mean(l2_errors),np.std(l2_errors)
    print ("L2 Difference in latent weights between instances: {}".format(np.sum((full_task_weights[0]-full_task_weights[1])**2)))
    
    def get_random_sample(start,stop,size):
        indices_set = set()
        while len(indices_set) < size:
            indices_set.add(np.random.randint(start,stop))
        return np.array(list(indices_set))
    
    # size of sample to compute error on
    sample_size = 10000
    for i in xrange(40):    
        # Update BNN network weights
        network.fit_network(exp_buffer_np, full_task_weights, 0, state_diffs=state_diffs,
                            use_all_exp=True)
        print('finished BNN update '+str(i))
        if i % 4 == 0:
            #get random sample of indices
            sample_indices = get_random_sample(0,X.shape[0],sample_size)
            l2_errors = network.get_td_error(np.hstack((X[sample_indices],full_task_weights[inst_indices[sample_indices]])), y[sample_indices], location=0.0, scale=1.0, by_dim=False)
            print ("After BNN update: iter: {}, Mean Error: {}, Std Error: {}".format(i,np.mean(l2_errors),np.std(l2_errors)))
        # Update latent weights
        for inst in np.random.permutation(batch_generator_hipmdp.instance_count):
            full_task_weights[inst,:] = network.optimize_latent_weighting_stochastic(
                exp_dict[inst],np.atleast_2d(full_task_weights[inst,:]),0,state_diffs=state_diffs,use_all_exp=True)
        print ('finished wb update '+str(i))
        # Compute error on sample of transitions
        if i % 4 == 0:
            #get random sample of indices
            sample_indices = get_random_sample(0,X.shape[0],sample_size)
            l2_errors = network.get_td_error(np.hstack((X[sample_indices],full_task_weights[inst_indices[sample_indices]])), y[sample_indices], location=0.0, scale=1.0, by_dim=False)
            print ("After Latent update: iter: {}, Mean Error: {}, Std Error: {}".format(i,np.mean(l2_errors),np.std(l2_errors)))
            # We check to see if the latent updates are sufficiently different so as to avoid fitting [erroneously] to the same dynamics
            print ("L2 Difference in latent weights between instances: {}".format(np.sum((full_task_weights[0]-full_task_weights[1])**2)))
    
    with open("{}/time.txt".format(dir_name), 'a') as f:
        f.write("%.5e" % (time.time() - t_start))
    
    network_weights = network.weights
    
    with open('{}/{}_network_weights'.format(dir_name, domain), 'w') as f:
        pickle.dump(network.weights, f)

if __name__ == "__main__":

    with open('config.json', 'r') as f:
        config = json.load(f)
    train_function(config)
