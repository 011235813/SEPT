import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


def actor_probe(state, n_h1=400, n_h2=200, n_h3=200, n_actions=5):
    """
    Actor network for probe policy
    Discrete actions
    """
    h1 = tf.layers.dense(inputs=state, units=n_h1, activation=tf.nn.relu, use_bias=True, name='probe_h1')
    h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu, use_bias=True, name='probe_h2')
    h3 = tf.layers.dense(inputs=h2, units=n_h3, activation=tf.nn.relu, use_bias=True, name='probe_h3')
    out = tf.layers.dense(inputs=h3, units=n_actions, activation=None, use_bias=True, name='probe_out')
    probs = tf.nn.softmax(out, name='probe_softmax')

    return probs


def actor_probe_cas(state, n_h1=400, n_h2=300, n_h3=200, n_actions=5, activation='relu'):

    if activation == 'relu':
        activation = tf.nn.relu
    elif activation == 'tanh':
        activation = tf.nn.tanh
    else:
        activation = tf.nn.relu

    h1 = tf.layers.dense(inputs=state, units=n_h1, activation=activation, use_bias=True, name='actor_h1')
    h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=activation, use_bias=True, name='actor_h2')
    mu = tf.layers.dense(inputs=h2, units=n_actions, activation=None, use_bias=True)
    logvar = tf.layers.dense(inputs=h2, units=n_actions, activation=None, use_bias=True)

    return mu, logvar


def encoder(x, n_h1, n_z):
    """
    For original VAE
    """
    h = tf.layers.dense(inputs=x, units=n_h1, activation=tf.nn.relu, use_bias=True)
    mu = tf.layers.dense(inputs=h, units=n_z, activation=None, use_bias=True)
    log_var = tf.layers.dense(inputs=h, units=n_z, activation=None, use_bias=True)
    
    return mu, log_var


def encoder_bidirectional_LSTM(trajs, timesteps, n_h=300, n_z=8):
    """
    Used for my alg

    trajs - shape is (batch_size, timesteps, state+action dim)
    n_h - number of hidden units, same for both forward and backward
    n_z - dimension of latent variable
    """
    trajs = tf.unstack(trajs, timesteps, 1)
    # lstm_forward_cell = rnn.BasicLSTMCell(n_h, forget_bias=1.0)
    lstm_forward_cell = rnn.LSTMCell(num_units=n_h, forget_bias=1.0)
    lstm_backward_cell = rnn.LSTMCell(num_units=n_h, forget_bias=1.0)
                               
    rnn_out, state_forward, state_backward = rnn.static_bidirectional_rnn(lstm_forward_cell, lstm_backward_cell, trajs, dtype=tf.float32)

    # Mean-pool over time
    rnn_mean = tf.reduce_mean(rnn_out, axis=0)

    mu = tf.layers.dense(inputs=rnn_mean, units=n_z, activation=None, use_bias=True)
    log_var = tf.layers.dense(inputs=rnn_mean, units=n_z, activation=None, use_bias=True)

    return mu, log_var


def decoder(z, n_h1, n_x):
    """
    For original VAE, with multivariate Bernoulli output
    """
    h = tf.layers.dense(inputs=z, units=n_h1, activation=tf.nn.relu, use_bias=True)
    y_logit = tf.layers.dense(inputs=h, units=n_x, activation=None, use_bias=True)
    y = tf.sigmoid(y_logit)

    return y_logit, y


def decoder_rnn(z, timesteps, n_x, batch_size, n_h=256, initial_input=None):
    """
    Used for my alg

    z - latent variable produced by encoder_bidirectional_LSTM, (batch_size, latent_dim)
    timesteps - length of trajectory (number of state-action pairs) to generate
    n_x - dimension of state + dimension of action
    batch_size - tf.placeholder
    n_h - number of hidden units
    initial_input - optional initial placeholder to act as (s_0, a_0)
    """
    # batch_size = z.shape.as_list()[0]
    if initial_input is None:
        initial_input = tf.zeros( [batch_size, n_x] )

    # lstm_cell = rnn.BasicLSTMCell(n_h, forget_bias=1.0)
    lstm_cell = rnn.LSTMCell(num_units=n_h, forget_bias=1.0)
    hidden_state = lstm_cell.zero_state(batch_size, tf.float32)

    W_mu = tf.get_variable('W_mu', dtype=tf.float32, shape=[n_h, n_x])
    b_mu = tf.get_variable('b_mu', dtype=tf.float32, shape=[n_x])

    W_logvar = tf.get_variable('W_logvar', dtype=tf.float32, shape=[n_h, n_x])
    b_logvar = tf.get_variable('b_logvar', dtype=tf.float32, shape=[n_x])

    list_mu = []
    list_logvar = []

    x = initial_input # x is concatenation of (s,a)
    for t in range(timesteps):
        x = tf.concat([x, z], axis=1)
        x_next, hidden_state = lstm_cell(x, hidden_state)
        mu = tf.matmul(x_next, W_mu) + b_mu
        log_var = tf.matmul(x_next, W_logvar) + b_logvar
        x = mu

        list_mu.append(mu)
        list_logvar.append(log_var)

    return list_mu, list_logvar


def Q_net(state, n_h1, n_h2, n_actions):
    """
    Used as control policy by some baselines.
    Follows architecture of Killian et al. 2017
    """
    h1 = tf.layers.dense(inputs=state, units=n_h1, activation=tf.nn.relu, use_bias=True, name='Q_h1')
    h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu, use_bias=True, name='Q_h2')
    out = tf.layers.dense(inputs=h2, units=n_actions, activation=None, use_bias=True, name='Q_out')

    return out


def Q_net_z(state, z, n_h1, n_h2, n_actions):
    """
    Used as control policy by my method and direct policy transfer (Yao et al. 2018)
    """
    concated = tf.concat([state, z], axis=1)
    h1 = tf.layers.dense(inputs=concated, units=n_h1, activation=tf.nn.relu, use_bias=True, name='Q_h1')
    h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu, use_bias=True, name='Q_h2')
    out = tf.layers.dense(inputs=h2, units=n_actions, activation=None, use_bias=True, name='Q_out')

    return out


def Q_net_z2(state, z, n_h1, n_h2, n_actions):

    h1 = tf.layers.dense(inputs=state, units=n_h1, activation=tf.nn.relu, use_bias=True, name='Q_h1')
    h1_z = tf.layers.dense(inputs=z, units=32, activation=tf.nn.relu, use_bias=True, name='Q_h1_z')
    concated = tf.concat([h1, h1_z], axis=1)

    h2 = tf.layers.dense(inputs=concated, units=n_h2, activation=tf.nn.relu, use_bias=True, name='Q_h2')
    out = tf.layers.dense(inputs=h2, units=n_actions, activation=None, use_bias=True, name='Q_out')

    return out    


def Q_net_zeta(state, z, eta, n_h1, n_h2, n_actions):
    """
    z - estimated or real hidden parameter value
    eta - uncertainty
    """
    concated = tf.concat([state, z, eta], axis=1)
    h1 = tf.layers.dense(inputs=concated, units=n_h1, activation=tf.nn.relu, use_bias=True, name='Q_h1')
    h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu, use_bias=True, name='Q_h2')
    out = tf.layers.dense(inputs=h2, units=n_actions, activation=None, use_bias=True, name='Q_out')

    return out
