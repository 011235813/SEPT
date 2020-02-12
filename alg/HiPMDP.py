"""Main environment wrapper.

Supports 2D navigation, Acrobot and HIV.
Supports training, validation and test phases
"""

import numpy as np
import pickle
import random
import sys

sys.path.append('../hip-mdp-public')


class HiPMDP(object):

    def __init__(self, domain, config_domain, phase='train'):
        """
        domain - string "2D", "acrobot", "hiv"
        config_domain - dictionary
        phase - string 'train', 'validation', test'
        """
        self.domain = domain
        self.config_domain = config_domain
        self.phase = phase

        self.N_train_instances = self.config_domain['N_train_instances']

        self.N_validation_instances = self.config_domain['N_validation_instances']
        self.N_test_instances = self.config_domain['N_test_instances']
        if self.phase == 'validation':
            self.idx_test = self.N_train_instances # start after last training instance
        elif self.phase == 'test':
            self.idx_test = self.N_train_instances + self.N_validation_instances
        
        self.set_domain_hyperparams()

    def set_domain_hyperparams(self):

        self.standardize_rewards = False
        self.standardize_states = False
        if self.domain == '2D':
            from grid_simulator.grid import Grid as model
            self.grid_beta = self.config_domain['grid_beta']
            self.env = model(beta=self.grid_beta)
            self.max_steps_per_episode = 100
            self.preset_hidden_params = self.config_domain['preset_hidden_params']
        elif self.domain == 'acrobot':
            from acrobot_simulator.acrobot_py3 import Acrobot as model
            self.env = model()
            self.max_steps_per_episode = 400
            with open('../hip-mdp-public/preset_parameters/%s'%self.config_domain['params_filename'],'rb') as f:
                # the pickle file was saved from Python2
                self.preset_hidden_params = pickle.load(f, encoding='latin1')
        elif self.domain == 'hiv':
            from hiv_simulator.hiv import HIVTreatment as model
            self.env = model()
            self.max_steps_per_episode = 200 if self.phase == 'train' else 1000
            self.standardize_states = True
            self.standardize_rewards = True
            with open('../hip-mdp-public/preset_parameters/%s'%self.config_domain['params_filename'], 'rb') as f:
                self.preset_hidden_params = pickle.load(f, encoding='latin1')
        else:
            raise ValueError("HiPMDP.py : domain not recognized")

        self.num_actions = self.env.num_actions
        self.state_dim = len(self.env.observe())

        if self.standardize_rewards:
            self.load_reward_standardization()
        if self.standardize_states:
            self.load_state_standardization()
        
    def load_reward_standardization(self):
        """
        Load the reward mean and standard deviation.
        """
        with open('../hip-mdp-public/preset_parameters/%s'%self.config_domain['reward_std_filename'], 'rb') as f:
            self.rewards_standardization = pickle.load(f, encoding='latin1')

    def load_state_standardization(self):
        """
        Load the state mean and standard deviation.
        """
        with open('../hip-mdp-public/preset_parameters/%s'%self.config_domain['state_std_filename'], 'rb') as f:
            self.state_mean, self.state_std = pickle.load(f, encoding='latin1')

    def standardize_state(self, state):
        """
        Standardize and return the given state.
        """
        return (state-self.state_mean) / self.state_std

    def switch_instance(self, code=None):
        """
        Sample another instance
        """
        if self.phase == 'train' or self.domain == '2D':
            # choose instances randomly for training, and for all phases of 2D world
            self.idx = random.randint(0, self.N_train_instances-1)
        else:
            # use each instance once during validation and test
            self.idx = self.idx_test
            self.idx_test += 1
        self.instance_param_set = self.preset_hidden_params[ self.idx ]

        if self.standardize_rewards:
            self.reward_mean, self.reward_std = self.rewards_standardization[self.idx]

    def get_real_hidden_param(self):
        """
        Returns the hidden parameter of the current instance as np.array
        """
        if self.domain == '2D':
            if self.instance_param_set['latent_code'] == 1:
                z = np.array([1,0], dtype=np.float32)
            else:
                z = np.array([0,1], dtype=np.float32)
        elif self.domain == 'acrobot' or self.domain == 'hiv':
            z = np.array(list(self.instance_param_set.values()))
        elif self.domain == 'lander':
            z = self.env.gym_env.get_hp()
        
        return z

    def step(self, action):

        reward, state_next = self.env.perform_action(action, **self.instance_param_set)
        if self.standardize_rewards:
            reward = (reward - self.reward_mean) / self.reward_std
        if self.standardize_states:
            state_next = self.standardize_state(state_next)

        done = self.env.is_done(self.max_steps_per_episode)

        if (self.domain == '2D') and (self.env.t == self.config_domain['t_switch']):
            # Switch dynamics once
            self.env.latent_code = 1 if self.env.latent_code == 2 else 2

        return state_next, reward, done

    def reset(self):

        self.env.reset(perturb_params=True, **self.instance_param_set)
        state = self.env.observe()
        if self.standardize_states:
            state = self.standardize_state(state)

        return state
