"""
Baseline 1: 
Training: train a single policy on all training instances, for many episodes per instance.
Test: execute the trained policy on the test instance without any further training.
"""

import sys, os
import time
import pickle
import json
import random

sys.path.append('../../hip-mdp-public/')
import ExperienceReplay

import numpy as np
import tensorflow as tf

import HiPMDP
import ddqn


def train_function(config, config_suffix=None):

    # with open('config.json') as f:
    #     config = json.load(f)
    config_main = config['main']
    config_DDQN = config['DDQN']
    config_PER = config['PER']
    config_baseline = config['baseline']
    
    real_z_input = config_baseline['real_z_input']
    if real_z_input:
        # If use real hidden param as input, then of course DDQN must accept z
        assert(config_DDQN['z_input']==True)
    
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
        raise ValueError("train_baseline.py : domain not recognized")
    with open(domain_name) as f:
        config_domain = json.load(f)
    
    n_state = config_domain['n_state']
    n_action = config_domain['n_action']
    n_hidden = config_domain['n_hidden'] # dimension of real hidden param
    
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
    
    # Instantiate control policy
    pi_c = ddqn.DDQN(config_DDQN, n_state, n_action, config_PER['activate'], n_hidden)
    epsilon_start = config_DDQN['epsilon_start']
    epsilon_end = config_DDQN['epsilon_end']
    epsilon_decay = np.exp(np.log(epsilon_end/epsilon_start)/(N_instances*N_episodes))
    # epsilon_decay = config_DDQN['epsilon_decay']
    steps_per_train = config_DDQN['steps_per_train']
    
    # TF session
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())
    
    sess.run(pi_c.list_initialize_target_ops)
    
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
    header = "Episode,R_avg\n"
    with open("../results/%s/log.csv" % dir_name, 'w') as f:
        f.write(header)
    reward_period = 0
    
    epsilon = epsilon_start
    control_step = 0
    train_count_control = 1
    total_episodes = 0
    t_start = time.time()
    # Iterate through random instances from the HPMDP
    for idx_instance in range(1, N_instances+1):
    
        hpmdp.switch_instance()
        print("idx_instance", idx_instance, " | Switching instance to", hpmdp.instance_param_set)
        if real_z_input:
            z = hpmdp.get_real_hidden_param()
    
        # Iterate through many episodes
        for idx_episode in range(1, N_episodes+1):
    
            total_episodes += 1
    
            # print("Episode", idx_episode)
            state = hpmdp.reset()
            done = False
            summarized = False
            reward_episode = 0
    
            # Start control policy
            while not done:
                # Use DDQN with prioritized replay for this
                if real_z_input:
                    action = pi_c.run_actor(state, z, sess, epsilon)
                else:
                    action = pi_c.run_actor(state, None, sess, epsilon)
                state_next, reward, done = hpmdp.step(action)
                control_step += 1
                reward_episode += reward
    
                if real_z_input:
                    buf.add(np.reshape(np.array([state,action,reward,state_next,done,z]), (1,6)))
                else:
                    buf.add(np.reshape(np.array([state,action,reward,state_next,done]), (1,5)))
                state = state_next
    
                if control_step >= min_samples_before_train and control_step % steps_per_train == 0:
                   batch, IS_weights, indices = buf.sample(control_step)
                   
                   if (total_episodes % period == 0) and not summarized:
                       # Write TF summary at first train step of the last episode of every instance
                       td_loss = pi_c.train_step(sess, batch, IS_weights, indices, train_count_control, summarize=True, writer=writer)
                       summarized = True
                   else:
                       td_loss = pi_c.train_step(sess, batch, IS_weights, indices, train_count_control, summarize=False, writer=writer)
                   train_count_control += 1
    
                   if config_PER['activate']:
                       buf.update_priorities( np.hstack( (np.reshape(td_loss, (len(td_loss),-1)), np.reshape(indices, (len(indices),-1))) ) )
    
            reward_period += reward_episode
    
            if epsilon > epsilon_end:
                epsilon *= epsilon_decay
    
            # Logging
            if total_episodes % period == 0:
                s = "%d,%.2f\n" % (total_episodes, reward_period/float(period))
                print(s)
                with open("../results/%s/log.csv" % dir_name, 'a') as f:
                    f.write(s)
                reward_period = 0
                
    with open("../results/%s/time.txt" % dir_name, 'a') as f:
        f.write("%.5e" % (time.time() - t_start))
    
    saver.save(sess, '../results/%s/%s' % (dir_name, model_name))

if __name__ == "__main__":

    with open('config.json', 'r') as f:
        config = json.load(f)
    train_function(config)    
