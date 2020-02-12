"""Train or test MAML."""

import sys, os
import time
import pickle
import json
import random

sys.path.append('../hip-mdp-public/')
import ExperienceReplay

import numpy as np
import tensorflow as tf

import ddqn_meta
import HiPMDP


def train_test_function(config, config_suffix=None):

    config_main = config['main']
    config_DDQN = config['DDQN']
    config_PER = config['PER']
    phase = config_main['phase']
    
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
    
    if phase == 'train':
        N_instances = config_main['N_instances']
        N_episodes = config_main['N_episodes']
        period = config_main['period']
    else:
        N_instances = config_domain['N_test_instances']
        N_episodes = config_domain['N_test_episodes']
        test_steps = config_domain['test_steps']
    dir_name = config_main['dir_name']
    model_name = config_main['model_name']
    
    if phase == 'train':
        os.makedirs('../results/%s'%dir_name, exist_ok=True)
    
    # Instantiate HPMDP
    hpmdp = HiPMDP.HiPMDP(domain, config_domain, phase)
    
    # Instantiate probe policy
    n_probe_steps = config_domain['traj_length']
    
    # Instantiate control policy
    pi_c = ddqn_meta.DDQN_meta(config_DDQN, n_state, n_action, config_PER['activate'], 0)
    if phase == 'train':
        epsilon_start = config_DDQN['epsilon_start']
        epsilon_end = config_DDQN['epsilon_end']
        epsilon_decay = np.exp(np.log(epsilon_end/epsilon_start)/(N_instances*N_episodes))
        steps_per_train = config_DDQN['steps_per_train']
    
    # TF session
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)
    if phase == 'train':
        sess.run(tf.global_variables_initializer())
        sess.run(pi_c.list_initialize_target_ops)
        epsilon = epsilon_start
        writer = tf.summary.FileWriter('../results/%s' % dir_name, sess.graph)
    else:
        epsilon = 0
    
    saver = tf.train.Saver()
    if phase != 'train':
        print("Restoring variables from %s" % dir_name)
        saver.restore(sess, '../results/%s/%s' % (dir_name, model_name))    
    
    if phase == 'train':
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
    
        control_step = 0
        train_count_control = 1
        total_episodes = 0
        t_start = time.time()
    else:
        reward_total = 0
        cumulative_reward = np.zeros((test_steps, N_instances))
        list_times = []    
    # Iterate through random instances from the HPMDP
    for idx_instance in range(1, N_instances+1):
    
        hpmdp.switch_instance()
        print("idx_instance", idx_instance, " | Switching instance to", hpmdp.instance_param_set)
    
        if phase != 'train':
            t_start = time.time()
        # Iterate through many episodes
        for idx_episode in range(1, N_episodes+1):
    
            if phase == 'train':
                total_episodes += 1
                count_train_steps = 0
    
            # list of (s,a,r,s',d) pairs
            buffer_adapt = [] # trajectory for adaptation
            state = hpmdp.reset()
            episode_step = 0
            done = False
            reward_episode = 0
    
            # Generate transitions for single gradient update
            traj_finished_early = False
            for step in range(1, n_probe_steps+1):
    
                action = pi_c.run_actor(state, None, sess, epsilon)
                state_next, reward, done = hpmdp.step(action)
    
                buffer_adapt.append(np.array([state,action,reward,state_next,done]))
                if phase == 'train':
                    buf.add(np.reshape(np.array([state,action,reward,state_next,done]), (1,5)))
    
                state = state_next
                reward_episode += reward
                if phase != 'train':
                    cumulative_reward[episode_step, idx_instance-1] = reward_episode
                    episode_step += 1
    
                if done and step < n_probe_steps:
                    traj_finished_early = True
                    print("train_meta.py : done is True while generating probe trajectory")
                    break
    
            if traj_finished_early:
                # Skip over rest of episode if finished early
                continue
    
            if phase == 'train' and total_episodes % period == 0:
                summarize = True
            else:
                summarize = False
    
            # Compute one gradient step using data from buffer_adapt
            # Same operation during both training and test
            pi_c.adapt_step(sess, np.array(buffer_adapt))
    
            # Run the rest of the episode using policy with the updated (primed) parameter
            summarized = False
            while not done:
                action = pi_c.run_actor_prime(state, None, sess, epsilon)
                state_next, reward, done = hpmdp.step(action)
                reward_episode += reward
                if phase == 'train':
                    buf.add(np.reshape(np.array([state,action,reward,state_next,done]), (1,5)))
                    control_step += 1
                else:
                    cumulative_reward[episode_step, idx_instance-1] = reward_episode
                    episode_step += 1
                    if episode_step >= test_steps:
                        break
                state = state_next
            
                if phase == 'train' and control_step >= min_samples_before_train and control_step % steps_per_train == 0:
                    count_train_steps += 1
    
            if phase == 'train':
                # Do the same number of train steps as regular DDQN would have done during the episode
                for idx_train in range(count_train_steps):
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
                
                if epsilon > epsilon_end:
                    epsilon *= epsilon_decay
                
                # Logging
                if total_episodes % period == 0:
                    s = "%d,%.2f\n" % (total_episodes, reward_period/float(period))
                    print(s)
                    with open("../results/%s/log.csv" % dir_name, 'a') as f:
                        f.write(s)
                    reward_period = 0
            else:
                if episode_step < test_steps:
                    remaining = np.ones(test_steps-episode_step) * reward_episode
                    cumulative_reward[episode_step:, idx_instance-1] = remaining
                reward_total += reward_episode
    
        if phase != 'train':
            list_times.append( time.time() - t_start )
                
    # Final logging
    if phase == 'train':
        with open("../results/%s/time.txt" % dir_name, 'a') as f:
            f.write("%.5e" % (time.time() - t_start))
        saver.save(sess, '../results/%s/%s' % (dir_name, model_name))
    else:
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

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)
    train_test_function(config)
