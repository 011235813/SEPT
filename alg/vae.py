import tensorflow as tf
import numpy as np
import random

import networks


class VAE(object):

    def __init__(self, n_state, n_action, traj_length, batch_size=100, n_latent=8,
                 n_h_encoder=300, n_h_decoder=256, lr=0.001, num_epochs=100,
                 entropy_coeff=0, seed=123456, dual=False, tau=0.005, beta=1):
        """
        n_state - state dimension
        n_action - action dimension
        traj_length - number of state-action pairs
        """
        self.n_state = n_state
        self.n_action = n_action
        self.n_input = self.n_state + self.n_action
        self.traj_length = traj_length

        self.batch_size = batch_size
        self.n_latent = n_latent
        self.n_h_encoder = n_h_encoder
        self.n_h_decoder = n_h_decoder
        self.lr = lr
        self.num_epochs = num_epochs
        self.entropy_coeff = entropy_coeff
        self.seed = seed
        self.dual = dual
        self.tau = tau
        self.beta = beta

        self.create_networks()
        if self.dual:
            self.create_networks_2()
            self.list_equate_dual_ops, self.list_update_dual_ops = self.get_assign_target_ops()
        self.create_VAE_train_op()
        self.create_summary()

        self.saver = tf.train.Saver()

        
    def create_networks(self):
        """
        Builds computational graph
        """
        self.traj = tf.placeholder(dtype=tf.float32, shape=[None, self.traj_length, self.n_input])
        with tf.variable_scope("Encoder"):
            self.mu, self.log_var = networks.encoder_bidirectional_LSTM(self.traj, self.traj_length, self.n_h_encoder, self.n_latent)
        self.var = tf.exp(self.log_var)
        self.std = tf.sqrt(self.var)
        
        self.z = tf.random_normal(shape=[self.n_latent], mean=self.mu, stddev=self.std)
        self.initial_input = tf.placeholder(dtype=tf.float32, shape=[None, self.n_input])
        self.batch_size_ph = tf.placeholder(dtype=tf.int32, shape=[])

        # traj_length-1 because s_0 is given as initial_input
        with tf.variable_scope("Decoder"):
            # list of length (traj_length -1), each element has shape [batch_size, n_input]
            list_mu, list_logvar = networks.decoder_rnn(self.z, self.traj_length-1, self.n_input, self.batch_size_ph, self.n_h_decoder, self.initial_input)
        self.means = tf.stack(list_mu, axis=1) # [batch_size, timesteps-1, n_input]
        self.variances = tf.exp(tf.stack(list_logvar, axis=1))

        # Cannot use tf.random_normal, because it doesn't give pdf
        self.decoder_out = tf.distributions.Normal(loc=self.means, scale=tf.sqrt(self.variances))
        self.probs = self.decoder_out.prob(self.traj[:,1:,:])
        self.log_probs = tf.log(self.probs+1e-15)


    def create_networks_2(self):

        with tf.variable_scope("Encoder_dual"):
            self.mu_2, self.log_var_2 = networks.encoder_bidirectional_LSTM(self.traj, self.traj_length, self.n_h_encoder, self.n_latent)
        self.var_2 = tf.exp(self.log_var_2)
        self.std_2 = tf.sqrt(self.var_2)
        self.z_2 = tf.random_normal(shape=[self.n_latent], mean=self.mu_2, stddev=self.std_2)

        with tf.variable_scope("Decoder_dual"):
            list_mu_2, list_logvar_2 = networks.decoder_rnn(self.z_2, self.traj_length-1, self.n_input, self.batch_size_ph, self.n_h_decoder, self.initial_input)
        self.means_2 = tf.stack(list_mu_2, axis=1) # [batch_size, timesteps-1, n_input]
        self.variances_2 = tf.exp(tf.stack(list_logvar_2, axis=1))
        self.decoder_out_2 = tf.distributions.Normal(loc=self.means_2, scale=tf.sqrt(self.variances_2))
        self.probs_2 = self.decoder_out_2.prob(self.traj[:,1:,:])
        self.log_probs_2 = tf.log(self.probs_2 + 1e-15)

        self.regularizer_2 = self.beta * -0.5 * tf.reduce_sum(1 + self.log_var_2 - tf.square(self.mu_2) - self.var_2, 1)
        self.reconstruction_error_2 = -tf.reduce_sum( tf.reshape(self.log_probs_2, [-1, (self.traj_length-1)*self.n_input]), axis=1 )


    def get_assign_target_ops(self):

        # ops for equating main and target
        list_initial_ops = []
        # ops for slow update of target toward main
        list_update_ops = []

        # Encoder
        list_encoder = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Encoder/')
        map_name_encoder = {v.name.split('Encoder')[1] : v for v in list_encoder}
        list_encoder_2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Encoder_dual/')
        map_name_encoder_2 = {v.name.split('dual')[1] : v for v in list_encoder_2}
        if len(list_encoder) != len(list_encoder_2):
            raise ValueError("get_initialize_target_ops : lengths of Encoder and Encoder_2 do not match")
        for name, var in map_name_encoder.items():
            # assign value of main variable to target variable
            list_initial_ops.append( map_name_encoder_2[name].assign(var) )
        for name, var in map_name_encoder.items():
            # incremental update of target towards main
            list_update_ops.append( map_name_encoder_2[name].assign( self.tau*var + (1-self.tau)*map_name_encoder_2[name] ) )

        # Decoder
        list_decoder = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Decoder/')
        map_name_decoder = {v.name.split('Decoder')[1] : v for v in list_decoder}
        list_decoder_2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Decoder_dual/')
        map_name_decoder_2 = {v.name.split('dual')[1] : v for v in list_decoder_2}
        if len(list_decoder) != len(list_decoder_2):
            raise ValueError("get_initialize_target_ops : lengths of Decoder and Decoder_2 do not match")
        for name, var in map_name_decoder.items():
            # assign value of main variable to target variable
            list_initial_ops.append( map_name_decoder_2[name].assign(var) )
        for name, var in map_name_decoder.items():
            # incremental update of target towards main
            list_update_ops.append( map_name_decoder_2[name].assign( self.tau*var + (1-self.tau)*map_name_decoder_2[name] ) )


        return list_initial_ops, list_update_ops


    def create_VAE_train_op(self):
        """
        Constructs the VAE loss function
        """
        self.regularizer = self.beta * -0.5 * tf.reduce_sum(1 + self.log_var - tf.square(self.mu) - self.var, 1)
        self.reconstruction_error = -tf.reduce_sum( tf.reshape(self.log_probs, [-1, (self.traj_length-1)*self.n_input]), axis=1 )
        if self.entropy_coeff == 0:
            self.loss = tf.reduce_mean(self.regularizer + self.reconstruction_error)
        else:
            self.entropy = 0.5 * tf.reduce_sum(1 + self.log_var, 1) # ignore the constant term
            self.loss = tf.reduce_mean(self.regularizer + self.reconstruction_error + self.entropy_coeff * self.entropy)
        # self.loss_validation = tf.reduce_mean(self.regularizer + self.reconstruction_error)
        self.opt = tf.train.AdamOptimizer(self.lr)
        self.op = self.opt.minimize(self.loss)


    def create_summary(self):
        """
        Op for writing to Tensorboard
        """
        summaries = [tf.summary.scalar('vae_loss', self.loss)]
        # summary_valid = [tf.summary.scalar('loss_validation', self.loss_validation)]
        encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Encoder/')
        for v in encoder_variables:
            summaries.append(tf.summary.histogram(v.op.name, v))
        grads = self.opt.compute_gradients(self.loss, encoder_variables)
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name+'/gradient', grad))

        decoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Decoder/')
        for v in decoder_variables:
            summaries.append(tf.summary.histogram(v.op.name, v))
        grads = self.opt.compute_gradients(self.loss, decoder_variables)
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name+'/gradient', grad))  
            
        self.summary_op = tf.summary.merge(summaries)
        # self.summary_valid_op = tf.summary.merge(summary_valid)


    def compute_lower_bound(self, traj, sess):
        """
        Used by train.py for computing reward for probe policy
        # traj - list of [state action] vectors, each pair is already concatenated
        traj - list of (state, action) pairs
        """
        batch_size = 1

        initial_pair = traj[0]
        concated = np.concatenate([ initial_pair[0], initial_pair[1] ])
        initial_input = np.array([ concated ])
        traj = np.array([np.concatenate([pair[0], pair[1]]) for pair in traj]).reshape(self.traj_length, self.n_input)

        feed = {self.initial_input : initial_input, self.traj : np.array([traj]),
                self.batch_size_ph : 1}

        if self.dual:
            reg, rec_error = sess.run([self.regularizer_2, self.reconstruction_error_2], feed_dict=feed)
        else:
            reg, rec_error = sess.run([self.regularizer, self.reconstruction_error], feed_dict=feed)
        lowerbound = -(reg + rec_error)

        return lowerbound
        

    def train_step(self, sess, dataset, count, summarize=False, writer=None):
        """
        Trains VAE for some epochs using current dataset

        # dataset - list of trajectories, each traj is list of [state action] vectors
        dataset - list of trajectories, each traj is a list of (state, action) pairs
        """
        for epoch in range(1, self.num_epochs+1):
            batch = random.sample(dataset, self.batch_size)
            initials = np.vstack([np.concatenate([traj[0][0], traj[0][1]]) for traj in batch])
            # Convert to [batch, traj_length, state+action dimension]
            trajs = [ [ np.concatenate([pair[0], pair[1]]) for pair in traj ] for traj in batch ]
            trajs = np.array(trajs)
            feed = {self.initial_input : initials, self.traj : trajs,
                    self.batch_size_ph : self.batch_size}

            if summarize and epoch==self.num_epochs:
                summary, _ = sess.run([self.summary_op, self.op], feed_dict=feed)
                writer.add_summary(summary, count)
            else:
                _ = sess.run(self.op, feed_dict=feed)

        if self.dual:
            sess.run(self.list_update_dual_ops)


    def decode(self, sess, z):
        """
        TODO
        """
        pass


    def encode(self, sess, traj):
        """
        traj - list of (state, action_1hot) pairs
        Computes estimate of hidden parameter from a single trajectory
        """
        traj = np.array([np.concatenate([pair[0], pair[1]]) for pair in traj]).reshape(self.traj_length, self.n_input)
        feed = {self.traj : np.array([traj])}
        z = sess.run(self.z, feed_dict=feed)[0]

        return z


    def get_std(self, sess, traj):
        """
        traj - list of (state, action_1hot) pairs
        Computes self.std a single trajectory
        """
        traj = np.array([np.concatenate([pair[0], pair[1]]) for pair in traj]).reshape(self.traj_length, self.n_input)
        feed = {self.traj : np.array([traj])}
        std = sess.run(self.std, feed_dict=feed)[0]

        return std


    def save_model(self, session, dir_name, model_name):
        self.saver.save(session, '../results/%s/%s' % (dir_name, model_name))


    def restore_model(self, session, dir_name, model_name):
        self.saver.restore(session, '../results/%s/%s' % (dir_name, model_name))
