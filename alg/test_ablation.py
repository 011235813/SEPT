import sys, os
import time
import pickle
import json
import random

sys.path.append('../hip-mdp-public/')

import numpy as np
import tensorflow as tf

import HiPMDP
import probe
import vae as vae_import
import ddqn


def test_function(config, config_suffix=None):

    config_main = config['main']
    config_probe = config['probe']
    config_VAE = config['VAE']
    config_DDQN = config['DDQN']
    config_PER = config['PER']
    config_ablation = config['ablation']
    use_pi_e = config_ablation['use_pi_e']
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
        raise ValueError("test_ablation.py : domain not recognized")
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
    
    # Instantiate probe policy
    n_probe_steps = config_domain['traj_length']
    assert(n_probe_steps < test_steps)
    if use_pi_e:
        pi_e = probe.Probe(config_probe, n_state, n_action)
    else:
        # initial z
        z_avg = pickle.load(open('../results/%s/z_avg.p'%dir_name, 'rb'))
    
    # Instantiate VAE
    buffer_size_vae = config_VAE['buffer_size']
    batch_size_vae = config_VAE['batch_size']
    del config_VAE['buffer_size']
    vae = vae_import.VAE(n_state, n_action, n_probe_steps, seed=seed, **config_VAE)
    
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
    # Iterate through random instances from the HPMDP
    for idx_instance in range(1, N_instances+1):
    
        hpmdp.switch_instance()
        print("idx_instance", idx_instance, " | Switching instance to", hpmdp.instance_param_set)
    
        # N_episodes should be 1, but we let it be flexible in case needed
        for idx_episode in range(1, N_episodes+1):
    
            reward_episode = 0
            
            collected_probe_traj = False
            while not collected_probe_traj:
    
                # list of (state, action) pairs
                traj_probe = []
                state = hpmdp.reset()
                episode_step = 0
                done = False
    
                probe_finished_early = False
                # Generate probe trajectory
                for step in range(1, n_probe_steps+1):
    
                    if use_pi_e:
                        action = pi_e.run_actor(state, sess)
                    else:
                        action = pi_c.run_actor(state, z_avg, sess, epsilon=0)
                    # print("Probe step %d action %d" % (step, action))
                    action_1hot = np.zeros(n_action)
                    action_1hot[action] = 1
                    traj_probe.append( (state, action_1hot) )
                    state_next, reward, done = hpmdp.step(action)
                    reward_episode += reward
                    cumulative_reward[episode_step, idx_instance-1] = reward_episode
                    state = state_next
                    episode_step += 1
                    if done and step < n_probe_steps:
                        probe_finished_early = True
                        print("test_ablation.py : done is True while generating probe trajectory")
                        break
    
                if not probe_finished_early:
                    collected_probe_traj = True
    
            # Use VAE to estimate hidden parameter
            z = vae.encode(sess, traj_probe)
    
            print(z)
    
            if config_DDQN['activate']:
                # Start control policy
                while not done and episode_step < test_steps:
                    # Use DDQN with prioritized replay for this
                    action = pi_c.run_actor(state, z, sess, epsilon=0)
                    state_next, reward, done = hpmdp.step(action)
                    reward_episode += reward
                    cumulative_reward[episode_step, idx_instance-1] = reward_episode
                    state = state_next
                    episode_step += 1
                print(reward_episode)
                # If episode ended earlier than test_steps, fill in the
                # rest of the cumulative rewards with the last value
                if episode_step < test_steps:
                    remaining = np.ones(test_steps-episode_step) * reward_episode
                    cumulative_reward[episode_step:, idx_instance-1] = remaining
    
                reward_total += reward_episode
    
    header = 'Step'
    for idx in range(1, N_instances+1):
        header += ',R_%d' % idx
    indices = np.arange(1,test_steps+1).reshape(test_steps,1)
    concated = np.concatenate([indices, cumulative_reward], axis=1)
    save_loc = '_'.join(dir_name.split('_')[:-1])
    os.makedirs('../results/%s'%save_loc, exist_ok=True)
    run_number = dir_name.split('_')[-1]
    np.savetxt('../results/%s/test_%s.csv'%(save_loc,run_number), concated, delimiter=',', fmt='%.3e', header=header)
    
    print("Avg episode reward", reward_total/float(N_instances*N_episodes))

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
    test_function(config)    
