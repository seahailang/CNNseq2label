#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: model.py.py
@time: 2018/4/18 16:32
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import utils

FLAGS = tf.app.flags.FLAGS


class Model(object):
    def __init__(self,iterator,matrix,config):
        self.max_train_steps = config.max_train_steps
        self.val_steps = config.val_steps
        self.global_step = tf.train.get_or_create_global_step()
        self.ckpt = config.ckpt
        self.num_class = config.num_class
        self.batch_size = config.batch_size
        self.embedding_matrix = tf.get_variable('embedding_matrix',
                                                shape = matrix.shape,
                                                initializer=tf.constant_initializer(matrix),
                                                trainable=False)
        self.seqs,self.labels,self.seq_lens = iterator.get_next()
        self.labels = tf.one_hot(self.labels,depth=4)
        self.learning_rate = config.learning_rate

        self.max_len = config.max_len

        if config.optimizer =='adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif config.optimizer =='adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

    def build_graph(self):
        X = tf.nn.embedding_lookup(self.embedding_matrix,self.seqs)

        W = tf.get_variable('dot_attenton_kernel_1',shape=[200,200],dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer())

        # X = utils.dot_attention(X,X)
        X_ = utils.conject_dot_attention(X,X,W,self.seq_lens)

        X = tf.concat((X,X_),axis=-1)
        X = tf.expand_dims(X,axis=2)
        X1 = tf.layers.separable_conv2d(X,filters=100,kernel_size =(3,1),padding='same')
        X2 = tf.layers.separable_conv2d(X,filters=100,kernel_size=(3,1),padding='same',dilation_rate=2)
        X3 = tf.layers.separable_conv2d(X,filters=100,kernel_size=(5,1),padding='same')
        X4 = tf.layers.separable_conv2d(X,filters=100,kernel_size=(5,1),padding='same',dilation_rate=2)
        X5 = tf.layers.separable_conv2d(X,filters=100,kernel_size=(5,1),padding='same',dilation_rate=3)

        # X1 = tf.layers.conv1d(X, filters=100, kernel_size=3, padding='same')
        # X2 = tf.layers.conv1d(X, filters=100, kernel_size=3, padding='same', dilation_rate=2)
        # X3 = tf.layers.conv1d(X, filters=100, kernel_size=5, padding='same')
        # X4 = tf.layers.conv1d(X, filters=100, kernel_size=5, padding='same', dilation_rate=2)
        # X5 = tf.layers.conv1d(X, filters=100, kernel_size=5, padding='same', dilation_rate=3)
        X = tf.concat([X1,X2,X3,X4,X5],axis=-1)
        X = tf.squeeze(X,axis=2)
        X = tf.layers.conv1d(X,filters=200,kernel_size=5,padding='same')
        X = tf.layers.conv1d(X,filters=200,kernel_size=5,padding='same')

        W2 = tf.get_variable('dot_attenton_kernel_2', shape=[200, 200], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer())
        # X = utils.dot_attention(X,X)
        X_ = utils.conject_dot_attention(X,X,W2,self.seq_lens)
        X = tf.concat((X,X_),axis=-1)
        X = tf.layers.conv1d(X,filters=1000,kernel_size=1,activation=tf.nn.relu)

        X = tf.layers.conv1d(X,filters=100,kernel_size=1,activation=tf.nn.relu)
        X = tf.layers.conv1d(X,filters=self.num_class,kernel_size=1,activation=None)
        self.logits = X
        return X


    def mask_loss(self):
        mask = tf.sequence_mask(self.seq_lens,maxlen=self.max_len)
        mask2 = tf.greater(tf.argmax(self.labels,axis=-1),0)
        mask3 = tf.greater(tf.random_uniform(shape=[self.batch_size,self.max_len],minval=0,maxval=1),0.8)
        mask = tf.logical_and(mask,tf.logical_or(mask2,mask3))
        labels = tf.boolean_mask(self.labels,mask)
        logits = tf.boolean_mask(self.logits,mask)
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        losses = tf.reduce_mean(losses)
        self.losses = losses
        return losses

    def loss(self):
        mask = tf.sequence_mask(self.seq_lens,maxlen=self.max_len)
        # mask2 = tf.greater(tf.argmax(self.labels,axis=-1),0)
        # mask3 = tf.greater(tf.random_uniform(shape=[self.batch_size,self.max_len],minval=0,maxval=1),0.2)
        # mask = tf.logical_and(mask,tf.logical_or(mask2,mask3))
        labels = tf.boolean_mask(self.labels,mask)
        logits = tf.boolean_mask(self.logits,mask)
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        losses = tf.reduce_mean(losses)
        self.losses = losses
        return losses

    def compute_gradients(self,loss,val_list=None):
        return self.optimizer.compute_gradients(loss,val_list)

    def apply_gradients(self,grads_and_vars,global_step=None):
        return self.optimizer.apply_gradients(grads_and_vars,global_step)








if __name__ == '__main__':
    pass