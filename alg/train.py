"""Trains SEPT and some ablations.

Ablations include TotalVar and MaxEnt.
"""

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
import probe
import vae as vae_import
import ddqn


def train_function(config, config_suffix=None):

    config_main = config['main']
    config_probe = config['probe']
    autoencoder = config_main['autoencoder']
    if autoencoder == 'VAE':
        config_VAE = config['VAE']
    else:
        raise ValueError("Other autoencoders not supported")
    config_DDQN = config['DDQN']
    config_PER = config['PER']
    phase = config_main['phase']
    assert(phase == 'train')
    
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
    elif domain == 'lander':
        domain_name = "config_lander.json"
    elif domain == 'cancer':
        domain_name = "config_cancer.json"
    else:
        raise ValueError("train.py : domain not recognized")
    with open(domain_name) as f:
        config_domain = json.load(f)
    
    n_state = config_domain['n_state']
    n_action = config_domain['n_action']
    min_samples_before_train = config_domain['min_samples_before_train']
    
    seed = config_main['seed']
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    
    N_instances = config_main['N_instances']
    N_episodes = config_main['N_episodes']
    period = config_main['period']
    dir_name = config_main['dir_name']
    model_name = config_main['model_name']
    
    os.makedirs('../results/%s'%dir_name, exist_ok=True)
    
    # Instantiate HPMDP
    hpmdp = HiPMDP.HiPMDP(domain, config_domain)
    
    # Instantiate probe policy
    n_probe_steps = config_domain['traj_length']
    pi_e = probe.Probe(config_probe, n_state, n_action)
    
    # Instantiate VAE
    buffer_size_vae = config_VAE['buffer_size']
    batch_size_vae = config_VAE['batch_size']
    del config_VAE['buffer_size']
    if autoencoder == 'VAE':
        vae = vae_import.VAE(n_state, n_action, n_probe_steps, seed=seed, **config_VAE)
    else:
        raise ValueError('Other autoencoders not supported')
    
    # Instantiate control policy
    if config_DDQN['activate']:
        pi_c = ddqn.DDQN(config_DDQN, n_state, n_action, config_PER['activate'], config_VAE['n_latent'])
        epsilon_start = config_DDQN['epsilon_start']
        epsilon_end = config_DDQN['epsilon_end']
        epsilon_decay = np.exp(np.log(epsilon_end/epsilon_start)/(N_instances*N_episodes))
        steps_per_train = config_DDQN['steps_per_train']
    
    # TF session
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())
    
    if config_DDQN['activate']:
        sess.run(pi_c.list_initialize_target_ops)
        epsilon = epsilon_start
    
    if config_VAE['dual']:
        sess.run(vae.list_equate_dual_ops)
    
    writer = tf.summary.FileWriter('../results/%s' % dir_name, sess.graph)
    
    saver = tf.train.Saver()
    
    # use the DQN version of the replay, so instance_count and bnn-specific params do not matter
    exp_replay_param = {'episode_count':N_instances*N_episodes,
                        'instance_count':0, 'max_task_examples':hpmdp.max_steps_per_episode,
                        'ddqn_batch_size':config_DDQN['batch_size'],
                        'num_strata_samples':config_PER['num_strata_samples'],
                        'PER_alpha':config_PER['alpha'],
                        'PER_beta_zero':config_PER['beta_zero'],
                        'bnn_batch_size':0, 'bnn_start':0,
                        'dqn_start':min_samples_before_train}
                        
    buf = ExperienceReplay.ExperienceReplay(exp_replay_param, buffer_size=config_PER['buffer_size'])
    
    # Logging
    header = "Episode,R_avg,R_p\n"
    with open("../results/%s/log.csv" % dir_name, 'w') as f:
        f.write(header)
    reward_period = 0
    reward_p_period = 0
    
    list_trajs = [] # circular buffer to store probe trajectories for VAE
    idx_traj = 0 # counter for list_trajs
    control_step = 0
    train_count_probe = 1
    train_count_vae = 1
    train_count_control = 1
    total_episodes = 0
    t_start = time.time()
    # Iterate through random instances from the HPMDP
    for idx_instance in range(1, N_instances+1):
    
        hpmdp.switch_instance()
        print("idx_instance", idx_instance, " | Switching instance to", hpmdp.instance_param_set)
    
        # Iterate through many episodes
        for idx_episode in range(1, N_episodes+1):
    
            total_episodes += 1
    
            # list of (state, action) pairs
            traj_probe = []
            state = hpmdp.reset()
            done = False
            reward_episode = 0
    
            # Generate probe trajectory
            probe_finished_early = False
            for step in range(1, n_probe_steps+1):
    
                action = pi_e.run_actor(state, sess)
                action_1hot = np.zeros(n_action)
                action_1hot[action] = 1
                traj_probe.append( (state, action_1hot) )
                state_next, reward, done = hpmdp.step(action)
                state = state_next
                reward_episode += reward
    
                if done and step < n_probe_steps:
                    probe_finished_early = True
                    print("train.py : done is True while generating probe trajectory")
                    break
    
            if probe_finished_early:
                # Skip over pi_e and VAE training if probe finished early
                continue
    
            if idx_traj >= len(list_trajs):
                list_trajs.append( traj_probe )
            else:
                list_trajs[idx_traj] = traj_probe
            idx_traj = (idx_traj + 1) % buffer_size_vae
    
            # Compute probe reward using VAE
            if config_probe['reward'] == 'vae':
                reward_e = vae.compute_lower_bound(traj_probe, sess)
            elif config_probe['reward'] == 'total_variation':
                reward_e = pi_e.compute_reward(traj_probe)
            elif config_probe['reward'] == 'negvae':
                # this reward encourages maximizing entropy
                reward_e = -vae.compute_lower_bound(traj_probe, sess)
    
            # Write Tensorboard at the final episode of every instance
            if total_episodes % period == 0:
                summarize = True
            else:
                summarize = False
    
            # Train probe policy
            pi_e.train_step(sess, traj_probe, reward_e, train_count_probe, summarize, writer)
            train_count_probe += 1
    
            # Train VAE
            if len(list_trajs) >= batch_size_vae:
                vae.train_step(sess, list_trajs, train_count_vae, summarize, writer)
                train_count_vae += 1
    
            # Use VAE to estimate hidden parameter
            z = vae.encode(sess, traj_probe)
    
            if config_DDQN['activate']:
                # Start control policy
                summarized = False
                while not done:
                    # Use DDQN with prioritized replay for this
                    action = pi_c.run_actor(state, z, sess, epsilon)
                    state_next, reward, done = hpmdp.step(action)
                    control_step += 1
                    reward_episode += reward
                
                    buf.add(np.reshape(np.array([state,action,reward,state_next,done,z]), (1,6)))
                    state = state_next
                
                    if control_step >= min_samples_before_train and control_step % steps_per_train == 0:
                       batch, IS_weights, indices = buf.sample(control_step)
                       if not summarized:
                           # Write TF summary at first train step of the last episode of every instance
                           td_loss = pi_c.train_step(sess, batch, IS_weights, indices, train_count_control, summarize, writer)
                           summarized = True
                       else:
                           td_loss = pi_c.train_step(sess, batch, IS_weights, indices, train_count_control, False, writer)
                       train_count_control += 1
                
                       if config_PER['activate']:
                           buf.update_priorities( np.hstack( (np.reshape(td_loss, (len(td_loss),-1)), np.reshape(indices, (len(indices),-1))) ) )
                
                reward_period += reward_episode
                reward_p_period += reward_e
    
                if epsilon > epsilon_end:
                    epsilon *= epsilon_decay
    
                # Logging
                if total_episodes % period == 0:
                    s = "%d,%.2f,%.2f\n" % (total_episodes, reward_period/float(period), reward_p_period/float(period))
                    print(s)
                    with open("../results/%s/log.csv" % dir_name, 'a') as f:
                        f.write(s)
                    if config_domain['save_threshold'] and reward_period/float(period) > config_domain['save_threshold']:
                        saver.save(sess, '../results/%s/%s.%d' % (dir_name, model_name, total_episodes))                    
                    reward_period = 0
                    reward_p_period = 0
                
    with open("../results/%s/time.txt" % dir_name, 'a') as f:
        f.write("%.5e" % (time.time() - t_start))
    
    saver.save(sess, '../results/%s/%s' % (dir_name, model_name))

if __name__ == "__main__":

    with open('config.json', 'r') as f:
        config = json.load(f)
    train_function(config)
