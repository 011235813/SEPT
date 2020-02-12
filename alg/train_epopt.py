"""
Adaptation of EPOpt to the case of HiP-MDP constrained to a single test episode
https://openreview.net/forum?id=SyWvgP5el
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
import vae
import ddqn


def train_function(config, config_suffix=None):

    config_main = config['main']
    config_DDQN = config['DDQN']
    config_PER = config['PER']
    assert(config['baseline']['epopt']==True)
    assert(config_DDQN['activate']==True)
    
    n_epopt = config['epopt']['n_epopt']
    epsilon_epopt = config['epopt']['epsilon']
    
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
    min_samples_before_train = config_domain['min_samples_before_train']
    
    seed = config_main['seed']
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    
    N_instances = config_main['N_instances']
    N_episodes_per_instance = config_main['N_episodes']
    # Give EPOpt the same number of total experiences as other methods
    N_episodes = N_instances * N_episodes_per_instance 
    period = config_main['period']
    dir_name = config_main['dir_name']
    model_name = config_main['model_name']
    
    os.makedirs('../results/%s'%dir_name, exist_ok=True)
    
    # Instantiate HPMDP
    hpmdp = HiPMDP.HiPMDP(domain, config_domain)
    
    # Instantiate control policy
    pi_c = ddqn.DDQN(config_DDQN, n_state, n_action, config_PER['activate'], 0)
    epsilon_start = config_DDQN['epsilon_start']
    epsilon_end = config_DDQN['epsilon_end']
    epsilon_decay = np.exp(np.log(epsilon_end/epsilon_start)/(N_episodes))
    steps_per_train = config_DDQN['steps_per_train']
    
    # TF session
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())
    
    sess.run(pi_c.list_initialize_target_ops)
    epsilon = epsilon_start
    
    writer = tf.summary.FileWriter('../results/%s' % dir_name, sess.graph)
    
    saver = tf.train.Saver()
    
    # number of episodes that will be stored into replay buffer,
    # accounting for the epsilon-percentile filtering
    effective_num_episodes = int(N_episodes/2.0 + N_episodes/2.0 * epsilon_epopt)
    # use the DQN version of the replay, so instance_count and bnn-specific params do not matter
    exp_replay_param = {'episode_count':effective_num_episodes,
                        'instance_count':0, 'max_task_examples':hpmdp.max_steps_per_episode,
                        'ddqn_batch_size':config_DDQN['batch_size'],
                        'num_strata_samples':config_PER['num_strata_samples'],
                        'PER_alpha':config_PER['alpha'],
                        'PER_beta_zero':config_PER['beta_zero'],
                        'bnn_batch_size':0, 'bnn_start':0,
                        'dqn_start':min_samples_before_train}
                        
    buf = ExperienceReplay.ExperienceReplay(exp_replay_param, buffer_size=config_PER['buffer_size'])
    
    # Logging
    header = "Episode,R_avg\n"
    with open("../results/%s/log.csv" % dir_name, 'w') as f:
        f.write(header)
    reward_period = 0
    
    control_step = 0
    train_count_control = 1
    idx_episode = 0
    summarize = False
    t_start = time.time()
    # Each iteration is one EPOpt iteration
    while idx_episode < N_episodes:
    
        instance_rollouts = []
        instance_total_rewards = []
        # Number of training steps that would have been done by online DDQN
        # during all of the episodes experienced by EPOpt
        expected_train_steps = 0 
        # Collect many episodes, HPs are reset every episode
        for idx_rollout in range(1, n_epopt+1):
    
            # This increment of the counter for the outer while loop is intentional.
            # This ensures EPOpt experiences the same number of episodes as other methods
            idx_episode += 1
    
            hpmdp.switch_instance()
            state = hpmdp.reset()
            done = False
            reward_episode = 0
            traj = []
            count_train_steps = 0
    
            while not done:
                # Use DDQN with prioritized replay for this
                action = pi_c.run_actor(state, None, sess, epsilon)
                state_next, reward, done = hpmdp.step(action)
                control_step += 1
                reward_episode += reward
            
                traj.append(np.reshape(np.array([state,action,reward,state_next,done]), (1,5)))
    
                state = state_next
            
                if control_step >= min_samples_before_train and control_step % steps_per_train == 0:
                    count_train_steps += 1
            
            instance_rollouts.append( traj )
            instance_total_rewards.append( reward_episode )
            expected_train_steps += count_train_steps
            reward_period += reward_episode
    
            if epsilon > epsilon_end:
                epsilon *= epsilon_decay
    
            # Logging
            if idx_episode % period == 0:
                s = "%d,%.2f\n" % (idx_episode, reward_period/float(period))
                print(s)
                with open("../results/%s/log.csv" % dir_name, 'a') as f:
                    f.write(s)
                reward_period = 0
                summarize = True
    
        if idx_episode < int(N_episodes / 2.0):
            # transitions from all trajectories will be stored
            percentile = 1.0
        else:
            percentile = epsilon_epopt
    
        # Compute epsilon-percentile of cumulative reward
        # Only store transitions from trajectories in the lowest epsilon-percentile
        sorted_indices = np.argsort(instance_total_rewards)
        indices_selected = sorted_indices[0 : int(n_epopt * percentile)]
        for idx_selected in indices_selected:
            for transition in instance_rollouts[idx_selected]:
                buf.add(transition)
    
        # Start control policy
        for idx_train in range(expected_train_steps):
            batch, IS_weights, indices = buf.sample(control_step)
            # Write TF summary at first train step after generating rollouts in which period was crossed
            td_loss = pi_c.train_step(sess, batch, IS_weights, indices, train_count_control, summarize, writer)
            summarize = False
            train_count_control += 1
            
            if config_PER['activate']:
                buf.update_priorities( np.hstack( (np.reshape(td_loss, (len(td_loss),-1)), np.reshape(indices, (len(indices),-1))) ) )
    
    
    with open("../results/%s/time.txt" % dir_name, 'a') as f:
        f.write("%.5e" % (time.time() - t_start))
    
    saver.save(sess, '../results/%s/%s' % (dir_name, model_name))

if __name__ == "__main__":

    with open('config.json', 'r') as f:
        config = json.load(f)
    train_function(config)
