
import tensorflow as tf
import layers as lr


def gaussian_mlp_encoder(x, h_dim=512, z_dim=2, keep_prob=0.9, reuse=None):
    with tf.variable_scope('encoder', reuse=False) as scope:
        body = lr.full_connect(x, h_dim, name='fc1')
        body = tf.nn.elu(body, name='elu')
        body = tf.nn.dropout(body, keep_prob)

        body = lr.full_connect(body, h_dim, name='fc2')
        body = tf.nn.tanh(body, name='tanh') #
        body = tf.nn.dropout(body, keep_prob)

        mu = lr.full_connect(body, z_dim, name='fc3_mu')
        sigma = lr.full_connect(body, z_dim, name='fc3_sigma')

        # The mean parameter is unconstrained
        mean = mu
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + tf.nn.softplus(sigma)
        return mean, stddev

def bernoulli_mlp_decoder(z, x_dim, h_dim=512, keep_prob=0.9, reuse=None):
    with tf.variable_scope('decoder', reuse=reuse):
        body = lr.full_connect(z, h_dim, name='fc1')
        body = tf.nn.tanh(body, name='tanh')
        body = tf.nn.dropout(body, keep_prob)

        body = lr.full_connect(body, h_dim, name='fc2')
        body = tf.nn.elu(body, name='elu')
        body = tf.nn.dropout(body, keep_prob)

        logits = lr.full_connect(body, x_dim, name='fc3')
        x = tf.nn.sigmoid(logits, name='sigmoid')
        x = tf.clip_by_value(x, 1e-8, 1 - 1e-8)

        return logits, x

def autoencoder(x, dim_img, dim_z, n_hidden, keep_prob):
    mu, sigma = gaussian_mlp_encoder(x, n_hidden, dim_z, keep_prob)
    z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
    logits, y = bernoulli_mlp_decoder(z, dim_img, n_hidden, keep_prob)
    y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)

    #
    marginal_likelihood = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits), 1)
    KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

    marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    KL_divergence = tf.reduce_mean(KL_divergence)
    loss = marginal_likelihood + KL_divergence

    return y, z, loss, marginal_likelihood, KL_divergence

def decoder(z, dim_img, n_hidden):
    _, y = bernoulli_mlp_decoder(z, dim_img, n_hidden, 1.0, reuse=True)
    return y

