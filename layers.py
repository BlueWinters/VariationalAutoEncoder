
import tensorflow as tf
import numpy as np


def full_connect(input, out_dim, name='fc'):
    with tf.variable_scope(name) as scope:
        in_dim = input.get_shape()[1]
        W = tf.get_variable(name='W', shape=[in_dim, out_dim], dtype=tf.float32,
                            initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable(name='b', shape=[out_dim], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.))
        a = tf.matmul(input, W) + b
    return a

    #tf.random_normal_initializer(stddev=0.1)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x, name)