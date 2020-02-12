"""Double DQN."""

import networks
import numpy as np
import tensorflow as tf


class DDQN(object):

    def __init__(self, config_ddqn, l_state, l_action, PER=False, n_latent=0):
        """
        PER - if true, activates prioritized replay with importance sampling correction
        n_latent - dimension of estimate of hidden parameter (only used if z_input = True)
        """
        self.l_state = l_state
        self.l_action = l_action
        self.l_z = n_latent

        self.z_input = config_ddqn['z_input']
        self.eta_input = config_ddqn['eta_input']
        self.n_h1 = config_ddqn['n_h1']
        self.n_h2 = config_ddqn['n_h2']

        self.batch_size = config_ddqn['batch_size']
        self.lr = config_ddqn['lr']
        self.clip = config_ddqn['clip']
        self.gamma = config_ddqn['gamma']
        self.tau = config_ddqn['tau']

        self.PER = PER

        self.create_networks()
        self.list_initialize_target_ops, self.list_update_target_ops = self.get_assign_target_ops()
        self.create_train_op()

        self.create_summary()

    def create_networks(self):

        self.state = tf.placeholder(tf.float32, [None, self.l_state], 'state')
        self.action_1hot = tf.placeholder(tf.float32, [None, self.l_action], 'action_1hot')

        if self.z_input and self.eta_input:
            self.z = tf.placeholder(tf.float32, [None, self.l_z], 'z')
            self.eta = tf.placeholder(tf.float32, [None, 1], 'eta')
            with tf.variable_scope("Q_main"):
                self.Q = networks.Q_net_zeta(self.state, self.z, self.eta, self.n_h1, self.n_h2, n_actions=self.l_action)
            with tf.variable_scope("Q_target"):
                self.Q_target = networks.Q_net_zeta(self.state, self.z, self.eta, self.n_h1, self.n_h2, n_actions=self.l_action)
        elif self.z_input:
            self.z = tf.placeholder(tf.float32, [None, self.l_z], 'z')
            with tf.variable_scope("Q_main"):
                self.Q = networks.Q_net_z(self.state, self.z, self.n_h1, self.n_h2, n_actions=self.l_action)
            with tf.variable_scope("Q_target"):
                self.Q_target = networks.Q_net_z(self.state, self.z, self.n_h1, self.n_h2, n_actions=self.l_action)
        else:
            with tf.variable_scope("Q_main"):
                self.Q = networks.Q_net(self.state, self.n_h1, self.n_h2, n_actions=self.l_action)
            with tf.variable_scope("Q_target"):
                self.Q_target = networks.Q_net(self.state, self.n_h1, self.n_h2, n_actions=self.l_action)

        self.argmax_Q = tf.argmax(self.Q, axis=1)

    def get_assign_target_ops(self):

        # ops for equating main and target
        list_initial_ops = []
        # ops for slow update of target toward main
        list_update_ops = []

        list_Q_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Q_main')
        map_name_Q_main = {v.name.split('main')[1] : v for v in list_Q_main}
        list_Q_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Q_target')
        map_name_Q_target = {v.name.split('target')[1] : v for v in list_Q_target}
        if len(list_Q_main) != len(list_Q_target):
            raise ValueError("get_initialize_target_ops : lengths of Q_main and Q_target do not match")
        for name, var in map_name_Q_main.items():
            # create op that assigns value of main variable to target variable of the same name
            list_initial_ops.append( map_name_Q_target[name].assign(var) )
        for name, var in map_name_Q_main.items():
            # incremental update of target towards main
            list_update_ops.append( map_name_Q_target[name].assign( self.tau*var + (1-self.tau)*map_name_Q_target[name] ) )

        return list_initial_ops, list_update_ops

    def run_actor(self, state, z, sess, epsilon=0, eta=None):
        """
        Runs policy using a single state as input
        z - estimate of the hidden parameter, used only if self.z_input = True
        eta - scalar representing uncertainty in estimated z, used only if self.eta_input==True
        """
        if np.random.rand(1) < epsilon:
            action = np.random.randint(0,self.l_action)
        else:
            feed = {self.state : np.array([state])}
            if self.z_input:
                feed[self.z] = np.array([z])
            if self.eta_input:
                feed[self.eta] = np.array([[eta]])
            action = sess.run(self.argmax_Q, feed_dict=feed)[0]
        
        return action

    def clip_if_not_none(self, gradient):

        if gradient is None:
            return gradient
        return tf.clip_by_norm(gradient, self.clip)

    def create_train_op(self):
        """
        Computes gradient descent using DDQN update rule with gradient clipping
        """
        self.td_target = tf.placeholder(tf.float32, [None], 'td_target')
        self.td_error = self.td_target - tf.reduce_sum(tf.multiply(self.Q, self.action_1hot), axis=1)
        self.td_loss = 0.5*tf.square(self.td_error)

        if self.PER:
            self.importance_weights = tf.placeholder(tf.float32, shape=[None])
            self.loss_Q = tf.reduce_mean(tf.multiply(self.importance_weights, self.td_loss))
        else:
            self.loss_Q = tf.reduce_mean(self.td_loss)

        if self.clip > 0:
            self.Q_opt = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=0.01)
            self.grads_and_vars = self.Q_opt.compute_gradients(self.loss_Q)
            self.clipped_grads = [ (self.clip_if_not_none(grad), var) for grad, var in self.grads_and_vars ]
            self.Q_op = self.Q_opt.apply_gradients(self.clipped_grads)
        else:
            self.Q_opt = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.Q_op = self.Q_opt.minimize(self.loss_Q)

    def create_summary(self):
        """
        Op for writing to Tensorboard
        """
        summaries_Q = [tf.summary.scalar('loss_Q', self.loss_Q)]
        Q_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Q_main')
        for v in Q_variables:
            summaries_Q.append( tf.summary.histogram(v.op.name, v) )
        grads = self.Q_opt.compute_gradients(self.loss_Q, Q_variables)
        for grad, var in grads:
            if grad is not None:
                summaries_Q.append( tf.summary.histogram(var.op.name+'/gradient', grad) )
        self.summary_op_Q = tf.summary.merge(summaries_Q)

    def process_actions(self, actions):
        """
        actions - [batch size]
        """
        n_steps = len(actions)
        actions_1hot = np.zeros( [n_steps, self.l_action], dtype=int )
        actions_1hot[ np.arange(n_steps), actions ] = 1

        return actions_1hot

    def process_batch(self, batch):

        if self.z_input and self.eta_input:
            states = np.stack(batch[:,0])
            z = np.stack(batch[:,1])
            eta = np.stack(batch[:,2])
            actions = np.stack(batch[:,3])
            rewards = np.stack(batch[:,4])
            states_next = np.stack(batch[:,5])
            z_next = np.stack(batch[:,6])
            eta_next = np.stack(batch[:,7])
            done = np.stack(batch[:,8])
            actions_1hot = self.process_actions(actions)
            return states, z, eta, actions_1hot, rewards, states_next, z_next, eta_next, done
        else:
            states = np.stack(batch[:,0])
            actions = np.stack(batch[:,1])
            rewards = np.stack(batch[:,2])
            states_next = np.stack(batch[:,3])
            done = np.stack(batch[:,4])
            actions_1hot = self.process_actions(actions)
            if self.z_input:
                z = np.stack(batch[:,5])
                return states, actions_1hot, rewards, states_next, done, z
            else:
                return states, actions_1hot, rewards, states_next, done

    def train_step(self, sess, batch, IS_weights, indices, count, summarize=False, writer=None):
        """
        One mini-batch update step.
        """
        if self.z_input and self.eta_input:
            states, z, eta, actions_1hot, rewards, states_next, z_next, eta_next, done = self.process_batch(batch)
            eta = np.reshape(eta, [-1,1])
            eta_next = np.reshape(eta_next, [-1,1])
            feed = {self.state : states_next, self.z : z, self.eta : eta}
            argmax_Q = sess.run(self.argmax_Q, feed_dict=feed)
            feed = {self.state : states_next, self.z : z_next, self.eta : eta_next}
            Q_target = sess.run(self.Q_target, feed_dict=feed)
        elif self.z_input:
            states, actions_1hot, rewards, states_next, done, z = self.process_batch(batch)
            feed = {self.state : states_next, self.z : z}
            argmax_Q = sess.run(self.argmax_Q, feed_dict=feed)
            feed = {self.state : states_next, self.z : z}
            Q_target = sess.run(self.Q_target, feed_dict=feed)            
        else:
            states, actions_1hot, rewards, states_next, done = self.process_batch(batch)
            feed = {self.state : states_next}
            argmax_Q = sess.run(self.argmax_Q, feed_dict=feed)
            feed = {self.state : states_next}
            Q_target = sess.run(self.Q_target, feed_dict=feed)            

        done_multiplier = -(done - 1)
        target = rewards + self.gamma * Q_target[range(self.batch_size), argmax_Q] * done_multiplier

        feed = {self.state : states,
                self.action_1hot : actions_1hot,
                self.td_target : target}
        if self.z_input:
            feed[self.z] = z
        if self.eta_input:
            feed[self.eta] = eta
        if self.PER:
            feed[self.importance_weights] = IS_weights

        if summarize:
            summary, _, td_loss = sess.run([self.summary_op_Q, self.Q_op, self.td_loss], feed_dict=feed)
            writer.add_summary(summary, count)
        else:
            _, td_loss = sess.run([self.Q_op, self.td_loss], feed_dict=feed)
        
        sess.run(self.list_update_target_ops)

        return td_loss
