import tensorflow as tf
import numpy as np
import networks


class Probe(object):

    def __init__(self, config_probe, l_state, l_action, action_space='discrete'):

        self.l_state = l_state
        self.l_action = l_action

        self.lr = config_probe['lr']
        self.n_h1 = config_probe['n_h1']
        self.n_h2 = config_probe['n_h2']
        self.n_h3 = config_probe['n_h3']
        self.activation = config_probe['activation']
        self.num_epochs = config_probe['num_epochs']

        self.action_space = action_space

        self.create_networks()
        self.create_policy_gradient_op()

        # TF summaries
        self.create_summary()


    def create_networks(self):

        self.state = tf.placeholder(tf.float32, [None, self.l_state], 'state')

        if self.action_space == 'discrete':
            with tf.variable_scope("Policy_probe"):
                self.probs = networks.actor_probe(self.state, self.n_h1, self.n_h2, self.n_h3, n_actions=self.l_action)
            # probs is normalized
            self.action_samples = tf.multinomial(tf.log(self.probs), 1)
        else:
            with tf.variable_scope("Policy_probe"):
                self.mu, self.logvar = networks.actor_probe_cas(self.state, self.n_h1, self.n_h2, self.n_h3, n_actions=self.l_action, activation=self.activation)
            self.variance = tf.exp(self.logvar)
            self.actor_dist = tf.distributions.Normal(loc=self.mu, scale=tf.sqrt(self.variance))
            self.action_samples = self.actor_dist.sample()


    def run_actor(self, state, sess, epsilon=0):

        feed = {self.state : np.array([state])}
        action = sess.run(self.action_samples, feed_dict=feed)
        if self.action_space == 'discrete':
            action = action[0][0]

        return action


    def compute_reward(self, traj):
        """
        traj - list of (state, action) pairs
        Returns average of absolute deviation over all state dimensions
        1/T sum_t sum_i | s_{t+1,i} - s_{t,i} |
        """
        T = len(traj)
        states = np.array( [pair[0] for pair in traj] )
        diff = states[1:,] - states[0:-1,]
        return np.array([(1/T) * np.sum(np.abs(diff))])


    def create_policy_gradient_op(self):
        """
        Computes policy gradient using trajectory reward
        """
        self.actions_taken = tf.placeholder(tf.float32, [None, self.l_action], 'action_taken')
        if self.action_space == 'discrete':
            # self.probs has shape [traj length, l_action]
            log_probs = tf.log(tf.reduce_sum(tf.multiply(self.probs, self.actions_taken), axis=1)+1e-15)
        else:
            # shape = (batch , l_action)
            log_probs_action_dims = tf.log( self.actor_dist.prob(self.actions_taken) + 1e-15 )
            # assume each action dimension is independent, given state
            log_probs = tf.reduce_sum( log_probs_action_dims, axis=1 )

        self.traj_reward = tf.placeholder(tf.float32, [1], 'traj_reward')
        self.probe_loss = - tf.reduce_sum(log_probs) * self.traj_reward
        self.policy_opt = tf.train.AdamOptimizer(self.lr)
        self.policy_op = self.policy_opt.minimize(self.probe_loss)


    def create_summary(self):
        """
        Op for writing to Tensorboard
        """
        summaries_policy = [tf.summary.scalar('probe_loss', self.probe_loss[0])]
        policy_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Policy_probe')
        for v in policy_variables:
            summaries_policy.append(tf.summary.histogram(v.op.name, v))
        grads = self.policy_opt.compute_gradients(self.probe_loss, policy_variables)
        for grad, var in grads:
            if grad is not None:
                summaries_policy.append( tf.summary.histogram(var.op.name+'/gradient', grad) )
        self.summary_op_policy = tf.summary.merge(summaries_policy)


    def process_actions(self, list_actions):
        
        n_steps = len(list_actions)
        actions_1hot = np.zeros( [n_steps, self.l_action], dtype=int )
        actions_1hot[ np.arange(n_steps), list_actions ] = 1

        return actions_1hot


    def train_step(self, sess, traj, reward, count, summarize=False, writer=None):
        """
        traj - list of (state, action) tuples
        reward - negative of the variational lower bound

        Called after each probe trajectory, when VAE has computed the lower bound
        """
        list_states = [ pair[0] for pair in traj ]
        list_actions = [ pair[1] for pair in traj ]

        feed = {self.actions_taken : list_actions,
                self.state : list_states,
                self.traj_reward : reward}

        for epoch in range(1, self.num_epochs+1):
            if summarize and epoch==self.num_epochs:
                summary, _ = sess.run([self.summary_op_policy, self.policy_op], feed_dict=feed)
                writer.add_summary(summary, count)
            else:
                _ = sess.run(self.policy_op, feed_dict=feed)
        
