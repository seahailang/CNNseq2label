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
        self.embedding_matrix = tf.get_variable('embedding_matrix',initializer=tf.constant_initializer(matrix))
        self.seqs,self.seq_lens,self.labels = iterator.get_next()
        self.learning_rate = config.learning_rate
        self.num_class = 4
        self.max_len = config.max_len

        if config.optimizer =='adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif config.optimizer =='adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

    def build_graph(self):
        X = tf.nn.embedding_lookup(self.embedding_matrix,self.seqs)
        X = utils.dot_attention(X,X)

        X1 = tf.layers.separable_conv1d(X,filters=100,kernel_size =3,padding='same')
        X2 = tf.layers.separable_conv1d(X,filters=100,kernel_size=3,padding='same',dilation_rate=2)
        X3 = tf.layers.separable_conv1d(X,filters=100,kernel_size=5,padding='same')
        X4 = tf.layers.separable_conv1d(X,filters=100,kernel_size=5,padding='same',dilation_rate=2)
        X5 = tf.layers.separable_conv1d(X,filters=100,kernel_size=5,padding='same',dilation_rate=3)
        X = tf.concat([X1,X2,X3,X4,X5],axis=-1)

        X = utils.dot_attention(X,X)

        X = tf.layers.conv1d(X,filters=1000,kernel_size=1,activation=tf.nn.relu)

        X = tf.layers.conv1d(X,filters=100,kernel_size=1,activation=tf.nn.relu)
        X = tf.layers.conv1d(X,filters=self.num_class,kernel_size=1,activation=None)
        self.logits = X
        return X


    def loss(self):
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels,logits=self.logits)
        mask = tf.sequence_mask(self.seq_lens,maxlen=self.max_len)
        losses = tf.boolean_mask(losses,mask)
        losses = tf.reduce_mean(losses)
        self.losses = losses
        return losses

    def compute_gradients(self,loss,val_list):
        return self.optimizer.compute_gradients(loss,val_list)

    def apply_gradients(self,grads_and_vars,global_step=None):
        return self.optimizer.apply_gradients(grads_and_vars,global_step)








if __name__ == '__main__':
    pass