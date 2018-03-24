
import tensorflow as tf
import numpy as np
import helper.tools as tools
import vae

from datetime import datetime
from helper.mnist import Mnist




def train():
    num_epochs = 500*100
    batch_size = 128
    learn_rate = 0.001
    x_dim = 28
    y_dim = 10
    z_dim = 2

    step_epochs = int(num_epochs/100)
    save_epochs = int(num_epochs/100)
    save_path = 'save/dim2'


    # In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
    x = tf.placeholder(tf.float32, shape=[None, x_dim*x_dim], name='x')
    z_in = tf.placeholder(tf.float32, shape=[None, z_dim], name='z')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')


    # encoder architecture
    mu, sigma = vae.gaussian_mlp_encoder(x, z_dim=z_dim, keep_prob=keep_prob)
    # sampling by re-parameterization technique
    z = mu + sigma * tf.random_normal(tf.shape(sigma), 0, 1, dtype=tf.float32)
    # decoder architecture
    logits, y = vae.bernoulli_mlp_decoder(z, x_dim=x_dim*x_dim, keep_prob=keep_prob)


    # reconstruction
    BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits), reduction_indices=1)
    AVE_BCE = tf.reduce_mean(BCE)
    # KL divergence
    KLD = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - 1 - tf.log(1e-9 + tf.square(sigma)), reduction_indices=1)
    AVE_KLD = tf.reduce_mean(KLD)
    # loss
    loss = AVE_BCE + AVE_KLD


    # optimization
    solver = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    # summary
    tf.summary.scalar('ELBO', loss)
    tf.summary.scalar('BCE', AVE_BCE)
    tf.summary.scalar('KLD', AVE_KLD)
    summary = tf.summary.merge_all()

    file = open('{}/train.txt'.format(save_path), 'w')
    sess = tf.Session()

    writer = tf.summary.FileWriter(logdir=save_path, graph=sess.graph)
    sess.run(tf.global_variables_initializer())

    # read data
    train_data = Mnist(train=True)
    test_data = Mnist(train=False)

    ave_loss_list = [0, 0, 0]
    min_loss = 1e10
    cur_time = datetime.now()
    z_sample = tools.get_mesh(z_range=2)

    toy_sample = test_data.images[:100]
    tools.save_grid_images(toy_sample, '{}/toy_input.png'.format(save_path), size=x_dim, nx=10, ny=10, chl=1)

    for epochs in range(1,1+num_epochs):
        batch_x, _ = train_data.next_batch(batch_size)

        sess.run([solver], feed_dict={x:batch_x, keep_prob:0.9})
        loss_list = sess.run([loss, AVE_BCE, AVE_KLD], feed_dict={x: batch_x, keep_prob: 0.9})
        tools.average_loss(ave_loss_list, loss_list, step_epochs)

        if epochs % step_epochs == 0:
            time_use = (datetime.now() - cur_time).seconds
            liner = "Epoch {:d}/{:d}, loss {:9f}, BCE {:9f}, KLD {:9f}, time_use {:f}" \
                .format(epochs, num_epochs, ave_loss_list[0], ave_loss_list[1], ave_loss_list[2], time_use)
            print(liner), file.writelines(liner + '\n')
            # summary
            step_summary = sess.run(summary, feed_dict={x:batch_x, keep_prob:1})
            writer.add_summary(step_summary, global_step=epochs)
            ave_loss_list = [0, 0, 0]  # reset to 0
            cur_time = datetime.now()

        step_loss = sess.run(loss, feed_dict={x:batch_x, keep_prob:1})

        if epochs % save_epochs == 0 and step_loss < min_loss:
            iter_counter = int(epochs / save_epochs)
            min_loss = step_loss
            x_toy_sample = sess.run(y, feed_dict={x:test_data.images[:100], keep_prob:1})
            tools.save_grid_images(x_toy_sample, '{}/toy_{}.png'.format(save_path, iter_counter), size=x_dim, nx=10, ny=10, chl=1)
            # only for when z_dim == 2 is true
            x_sample = sess.run(y, feed_dict={z:z_sample, keep_prob:1})
            tools.save_grid_images(x_sample, '{}/mainfold_{}.png'.format(save_path, iter_counter), size=x_dim, chl=1)
            train_z_map = sess.run(z, feed_dict={x:train_data.images[:5000], keep_prob:1})
            tools.save_scattered_image(train_z_map, train_data.labels[:5000], '{}/train_z_map_{}.png'.format(save_path, iter_counter), z_range=4)
            test_z_map = sess.run(z, feed_dict={x: test_data.images[:5000], keep_prob: 1})
            tools.save_scattered_image(test_z_map, test_data.labels[:5000], '{}/test_z_map_{}.png'.format(save_path, iter_counter), z_range=4)


    # save model
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(var_list=vars)
    saver.save(sess, save_path='{}/model'.format(save_path))

    # close all
    file.close()
    sess.close()




if __name__ == '__main__':
    train()
