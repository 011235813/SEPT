import sys, os
import time
import pickle
import json
import random

sys.path.append('../hip-mdp-public/')
import ExperienceReplay

import numpy as np
import tensorflow as tf

import HiPMDP
import vae as vae_import
import ddqn

def test_function(config, config_suffix=None):

    config_main = config['main']
    config_VAE = config['VAE']
    config_DDQN = config['DDQN']
    config_PER = config['PER']
    phase = config_main['phase']
    assert(phase == 'validation' or phase == 'test')
    
    domain = config_main['domain']
    
    # Domain-specific parameters (e.g. state and action space dimensions)
    if domain == '2D':
        domain_name = "config_2D.json"
    elif domain == 'acrobot':
        domain_name = "config_acrobot.json"
    elif domain == 'hiv':
        if config_suffix is not None:
            domain_name = "config_hiv{}.json".format(config_suffix)
        else:
            domain_name = "config_hiv.json"
    elif domain == 'mujoco':
        domain_name = "config_mujoco.json"
    elif domain == 'cancer':
        domain_name = "config_cancer.json"
    else:
        raise ValueError("train.py : domain not recognized")
    with open(domain_name) as f:
        config_domain = json.load(f)
    
    n_state = config_domain['n_state']
    n_action = config_domain['n_action']
    
    seed = config_main['seed']
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    
    N_instances = config_domain['N_test_instances']
    N_episodes = config_domain['N_test_episodes']
    test_steps = config_domain['test_steps']
    dir_name = config_main['dir_name']
    model_name = config_main['model_name']
    
    # Instantiate HPMDP
    hpmdp = HiPMDP.HiPMDP(domain, config_domain, phase)
    
    # Length of trajectory for input to VAE
    n_vae_steps  = config_domain['traj_length']
    n_latent = config_VAE['n_latent']
    z = np.zeros(config_VAE['n_latent'], dtype=np.float32)
    
    with open('../results/%s/std_max.pkl' % dir_name, 'rb') as f:
        std_max = pickle.load(f)
    
    # Instantiate VAE
    buffer_size_vae = config_VAE['buffer_size']
    batch_size_vae = config_VAE['batch_size']
    del config_VAE['buffer_size']
    vae = vae_import.VAE(n_state, n_action, n_vae_steps, seed=seed, **config_VAE)
    
    # Instantiate control policy
    if config_DDQN['activate']:
        pi_c = ddqn.DDQN(config_DDQN, n_state, n_action, config_PER['activate'], config_VAE['n_latent'])
    
    # TF session
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)
    
    saver = tf.train.Saver()
    print("Restoring variables from %s" % dir_name)
    saver.restore(sess, '../results/%s/%s' % (dir_name, model_name))
    
    reward_total = 0
    cumulative_reward = np.zeros((test_steps, N_instances))
    list_times = []
    # Iterate through random instances from the HPMDP
    for idx_instance in range(1, N_instances+1):
    
        hpmdp.switch_instance()
        print("idx_instance", idx_instance, " | Switching instance to", hpmdp.instance_param_set)
    
        t_start = time.time()
        for idx_episode in range(1, N_episodes+1):
    
            # rolling window of (state, action) pairs
            traj_for_vae = []
            eta = 1.0 # range [0,1] 1 means the policy should act to maximize probe reward
            z = np.zeros(config_VAE['n_latent'], dtype=np.float32)
            reward_episode = 0
            state = hpmdp.reset()
            episode_step = 0
            done = False
    
            while not done and episode_step < test_steps:
    
                action = pi_c.run_actor(state, z, sess, epsilon=0, eta=eta)
                action_1hot = np.zeros(n_action)
                action_1hot[action] = 1
                traj_for_vae.append( (state, action_1hot) )
                if len(traj_for_vae) == n_vae_steps + 1:
                    traj_for_vae = traj_for_vae[1:]
    
                state_next, reward, done = hpmdp.step(action)
    
                reward_episode += reward
                cumulative_reward[episode_step, idx_instance-1] = reward_episode
    
                # Get z_next and eta_next, because they are considered part of the augmented MDP state
                if len(traj_for_vae) == n_vae_steps:
                    std = vae.get_std(sess, traj_for_vae)
                    std = std / std_max # element-wise normalization, now each element is between [0,1]
                    eta_next = np.sum(std) / n_latent # scalar between [0,1]
                    eta_next = min(1.0, eta_next) # in case std_max during training isn't large enough
                    # Use VAE to update hidden parameter
                    z_next = vae.encode(sess, traj_for_vae)
                else:
                    z_next = z
                    eta_next = eta
    
                state = state_next
                eta = eta_next
                z = z_next
                episode_step += 1
    
            # If episode ended earlier than test_steps, fill in the
            # rest of the cumulative rewards with the last value
            if episode_step < test_steps:
                remaining = np.ones(test_steps-episode_step) * reward_episode
                cumulative_reward[episode_step:, idx_instance-1] = remaining
    
            reward_total += reward_episode
    
        list_times.append( time.time() - t_start )    
                
    header = 'Step'
    for idx in range(1, N_instances+1):
        header += ',R_%d' % idx
    indices = np.arange(1,test_steps+1).reshape(test_steps,1)
    concated = np.concatenate([indices, cumulative_reward], axis=1)
    save_loc = '_'.join(dir_name.split('_')[:-1])
    os.makedirs('../results/%s'%save_loc, exist_ok=True)
    run_number = dir_name.split('_')[-1]
    np.savetxt('../results/%s/test_%s.csv'%(save_loc,run_number), concated, delimiter=',', fmt='%.3e', header=header)
    
    with open('../results/%s/test_time_%s.pkl'%(save_loc,run_number), 'wb') as f:
        pickle.dump(list_times, f)
    
    print("Avg episode reward", reward_total/float(N_instances*N_episodes))

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
    test_function(config)
