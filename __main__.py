#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: __main__.py.py
@time: 2018/4/18 16:33
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from sklearn.model_selection import train_test_split


from model import Model
import utils
import datasets
from config import Config

FLAGS = tf.app.flags.FLAGS


def train(config):
    seq,label,seqlen,matrix = utils.load_data(config)
    train_seq,train_label,train_seq_len,\
    val_seq,val_label,val_seq_len =train_test_split(seq,label,seqlen,train_size=0.8,random_state=1234)
    train_data = (train_seq,train_label,train_seq_len)
    val_data = (val_seq,val_label,val_seq_len)
    train_dataset = datasets.make_train_data(*train_data)
    val_dataset = datasets.make_train_data(*val_data)
    train_iterator = train_dataset.make_one_shot_iterator()
    val_iterator = val_dataset.make_one_shot_iterator()

    holder = tf.placeholder(tf.string,[])
    iterator = tf.data.Iterator.from_string_handle(holder,output_types=train_iterator.output_types,
                                             output_shapes=train_iterator.output_shapes)

    model = Model(iterator=iterator,matrix=matrix,config=config)

    logits = model.build_graph()
    losses = model.loss()
    grads_and_vars = model.compute_gradients(losses,None)
    run_op = model.apply_gradients(grads_and_vars,model.global_step)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        train_handle = sess.run(train_iterator.string_handle())
        val_handl = sess.run(val_iterator.string_handle())
        ckpt = tf.train.latest_checkpoint(model.ckpt)
        if ckpt:
            saver.restore(sess,ckpt)
        for i in range(model.max_train_steps):
            g,l,_ = sess.run([model.global_step,losses,run_op],feed_dict={holder:train_handle})
            if g%model.val_steps==0:
                print(g,l)
                saver.save(sess,model.ckpt,g)



if __name__ == '__main__':
    config = Config()
    train(config)