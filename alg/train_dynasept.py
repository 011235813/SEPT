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

def train_function(config, config_suffix=None):

    config_main = config['main']
    config_VAE = config['VAE']
    config_DDQN = config['DDQN']
    config_PER = config['PER']
    config_ablation = config['ablation']
    eq_rew = config_ablation['equalize_reward']
    
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
    N_episodes = config_main['N_episodes']
    period = config_main['period']
    dir_name = config_main['dir_name']
    model_name = config_main['model_name']
    
    os.makedirs('../results/%s'%dir_name, exist_ok=True)
    
    # Instantiate HPMDP
    hpmdp = HiPMDP.HiPMDP(domain, config_domain)
    
    # Length of trajectory for input to VAE
    n_vae_steps  = config_domain['traj_length']
    n_latent = config_VAE['n_latent']
    z = np.zeros(config_VAE['n_latent'], dtype=np.float32)
    eta = 1.0 # range [0,1] 1 means the policy should act to maximize probe reward
    std_max = - np.inf * np.ones(config_VAE['n_latent'], dtype=np.float32)
    
    # Instantiate VAE
    buffer_size_vae = config_VAE['buffer_size']
    batch_size_vae = config_VAE['batch_size']
    del config_VAE['buffer_size']
    vae = vae_import.VAE(n_state, n_action, n_vae_steps, seed=seed, **config_VAE)
    
    
    # Instantiate control policy
    if config_DDQN['activate']:
        pi_c = ddqn.DDQN(config_DDQN, n_state, n_action, config_PER['activate'], config_VAE['n_latent'])
        epsilon_start = config_DDQN['epsilon_start']
        epsilon_end = config_DDQN['epsilon_end']
        epsilon_decay = np.exp(np.log(epsilon_end/epsilon_start)/(N_episodes*N_instances))
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
    
    # running mean and variance of MDP reward and VAE lowerbound
    if eq_rew:
        stat_counter = 0
        r_mdp_mean = 0
        r_mdp_var = 0
        r_probe_mean = 0
        r_probe_var = 0
    
    # Logging
    header = "Episode,R_avg,R_e_avg\n"
    with open("../results/%s/log.csv" % dir_name, 'w') as f:
        f.write(header)
    reward_period = 0
    reward_e_period = 0
    
    list_trajs = [] # circular buffer to store probe trajectories for VAE
    idx_traj = 0 # counter for list_trajs
    control_step = 0
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
    
            eta = 1.0
            z = np.zeros(config_VAE['n_latent'], dtype=np.float32)
            if total_episodes % period == 0:
                list_eta = [eta]
    
            # rolling window of (state, action) pairs
            traj_for_vae = []
            state = hpmdp.reset()
            done = False
            reward_episode = 0
            reward_e_episode = 0
            step_episode = 0
    
            if total_episodes % period == 0:
                summarize = True
            else:
                summarize = False
    
            summarized = False
            while not done:
    
                action = pi_c.run_actor(state, z, sess, epsilon, eta)
                control_step += 1
                action_1hot = np.zeros(n_action)
                action_1hot[action] = 1
                traj_for_vae.append( (state, action_1hot) )
                if len(traj_for_vae) == n_vae_steps + 1:
                    traj_for_vae = traj_for_vae[1:]
    
                state_next, reward, done = hpmdp.step(action)
                step_episode += 1
    
                if eq_rew:
                    stat_counter += 1
                    # update MDP reward mean and var
                    r_mdp_mean_prev = r_mdp_mean
                    r_mdp_mean = 1/float(stat_counter)*reward + (stat_counter-1)/float(stat_counter) * r_mdp_mean
                    r_mdp_var = r_mdp_var + (reward - r_mdp_mean_prev)*(reward - r_mdp_mean)
    
                if len(traj_for_vae) == n_vae_steps:
                    # Compute probe reward using VAE
                    reward_e = vae.compute_lower_bound(traj_for_vae, sess)[0]
    
                    if eq_rew:
                        # Update probe reward mean and var
                        r_probe_mean_prev = r_probe_mean
                        r_probe_mean = 1/float(stat_counter)*reward_e + (stat_counter-1)/float(stat_counter)*r_probe_mean
                        r_probe_var = r_probe_var + (reward_e - r_probe_mean_prev)*(reward_e - r_probe_mean)
                        # Scale probe reward into MDP reward
                        reward_e = ((reward_e - r_probe_mean)/np.sqrt(r_probe_var/stat_counter) + r_mdp_mean) * np.sqrt(r_mdp_var/stat_counter)
                    
                    reward_total = eta * reward_e + (1 - eta) * reward
                else:
                    reward_e = 0.0
                    reward_total = reward
    
                # Get z_next and eta_next, because they are considered part of the augmented MDP state
                if len(traj_for_vae) == n_vae_steps:
                    std = vae.get_std(sess, traj_for_vae)
                    # Update max
                    for idx in range(n_latent):
                        if std[idx] >= std_max[idx]:
                            std_max[idx] = std[idx]
                    std = std / std_max # element-wise normalization, now each element is between [0,1]
                    eta_next = np.sum(std) / n_latent # scalar between [0,1]
                    # Use VAE to update hidden parameter
                    z_next = vae.encode(sess, traj_for_vae)
                else:
                    z_next = z
                    eta_next = eta
    
                if total_episodes % period == 0:
                    list_eta.append(eta_next)
    
                # Use total reward to train policy
                buf.add(np.reshape(np.array([state,z,eta,action,reward_total,state_next,z_next,eta_next,done]), (1,9)))
                state = state_next
                eta = eta_next
                z = z_next
    
                # Note that for evaluation purpose we record the MDP reward separately
                reward_episode += reward
                reward_e_episode += reward_e
    
                # Store non-overlapping trajectories for training VAE
                # if len(traj_for_vae) == n_vae_steps:
                if step_episode % n_vae_steps == 0:
                    if idx_traj >= len(list_trajs):
                        list_trajs.append( list(traj_for_vae) ) # must make a new list
                    else:
                        list_trajs[idx_traj] = list(traj_for_vae)
                    idx_traj = (idx_traj + 1) % buffer_size_vae
    
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
            reward_e_period += reward_e_episode
    
            if epsilon > epsilon_end:
                epsilon *= epsilon_decay
    
            # Train VAE at the end of each episode
            if len(list_trajs) >= batch_size_vae:
                vae.train_step(sess, list_trajs, train_count_vae, summarize, writer)
                train_count_vae += 1
    
            # Logging
            if total_episodes % period == 0:
                s = "%d,%.2f,%.2f\n" % (total_episodes, reward_period/float(period), reward_e_period/float(period))
                print(s)
                with open("../results/%s/log.csv" % dir_name, 'a') as f:
                    f.write(s)
                with open("../results/%s/eta.csv" % dir_name, 'a') as f:
                    eta_string = ','.join(['%.2f' % x for x in list_eta])
                    eta_string += '\n'
                    f.write(eta_string)
                if config_domain['save_threshold'] and reward_period/float(period) > config_domain['save_threshold']:
                    saver.save(sess, '../results/%s/%s.%d' % (dir_name, model_name, total_episodes))
                reward_period = 0
                reward_e_period = 0
                
    with open("../results/%s/time.txt" % dir_name, 'a') as f:
        f.write("%.5e" % (time.time() - t_start))
    
    with open('../results/%s/std_max.pkl' % dir_name, 'wb') as f:
        pickle.dump(std_max, f)
    
    if eq_rew:
        reward_scaling = np.array([r_mdp_mean, np.sqrt(r_mdp_var/stat_counter), r_probe_mean, np.sqrt(r_probe_var/stat_counter)])
        with open('../results/%s/reward_scaling.pkl' % dir_name, 'wb') as f:
            pickle.dump(reward_scaling, f)
    
    saver.save(sess, '../results/%s/%s' % (dir_name, model_name))

if __name__ == "__main__":

    with open('config.json', 'r') as f:
        config = json.load(f)
    train_function(config)    
