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
from sklearn.metrics import accuracy_score,f1_score
import numpy as np


from model import Model
import utils
import datasets
from config import Config

FLAGS = tf.app.flags.FLAGS


def train(config):
    train_summary = []
    val_summary = []

    seq,label,seqlen,matrix = utils.load_data(config)
    train_seq,val_seq,train_label,val_label,train_seq_len,\
    val_seq_len =train_test_split(seq,label,seqlen,train_size=0.8,random_state=1234)
    train_data = (train_seq,train_label,train_seq_len)
    val_data = (val_seq,val_label,val_seq_len)
    train_dataset = datasets.make_train_data(*train_data)
    train_dataset = train_dataset.repeat().batch(config.batch_size)
    val_dataset = datasets.make_train_data(*val_data)
    val_dataset = val_dataset.repeat().batch(1)
    train_iterator = train_dataset.make_one_shot_iterator()
    val_iterator = val_dataset.make_one_shot_iterator()

    holder = tf.placeholder(tf.string,[])
    iterator = tf.data.Iterator.from_string_handle(holder,output_types=train_iterator.output_types,
                                             output_shapes=train_iterator.output_shapes)

    model = Model(iterator=iterator,matrix=matrix,config=config)

    logits = model.build_graph()
    losses = model.mask_loss()
    true_losses = model.loss()

    train_summary.append(tf.summary.scalar('train_loss',losses))

    val_losses_holder = tf.placeholder(tf.float32,[])
    val_acc_holder = tf.placeholder(tf.float32,[])
    val_f1_holder = tf.placeholder(tf.float32,[])
    val_summary.append(tf.summary.scalar('val_loss',val_losses_holder))
    val_summary.append(tf.summary.scalar('val_accuracy',val_acc_holder))
    val_summary.append(tf.summary.scalar('val_f1',val_f1_holder))

    grads_and_vars = model.compute_gradients(losses,None)
    run_op = model.apply_gradients(grads_and_vars,model.global_step)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    val_summary = tf.summary.merge(val_summary)
    train_summary = tf.summary.merge(train_summary)
    writer = tf.summary.FileWriter(model.ckpt)
    with tf.Session() as sess:
        sess.run(init)
        train_handle = sess.run(train_iterator.string_handle())
        val_handle = sess.run(val_iterator.string_handle())
        ckpt = tf.train.latest_checkpoint(model.ckpt)
        if ckpt:
            saver.restore(sess,ckpt)
            print('restore model from %s'%ckpt)
        for i in range(model.max_train_steps):
            g,l,s,_ = sess.run([model.global_step,losses,train_summary,run_op],feed_dict={holder:train_handle})
            if g%model.val_steps==0:
                print(g,l)
                writer.add_summary(s,g)
                saver.save(sess,model.ckpt,g)
                ground_truth = []
                predictions = []
                val_losses = []
                for j in range(len(val_seq)):
                        p,l,length,lo = sess.run([logits,model.labels,model.seq_lens,true_losses],feed_dict={holder:val_handle})
                        for line_p,line_l,line_length in zip(p,l,length):
                            line_p = np.argmax(line_p,axis=1)
                            line_l = np.argmax(line_l,axis=1)
                            for k in range(line_length):
                                if line_l[k]>0 or line_p[k]>0:
                                    ground_truth.append(line_l[k])
                                    predictions.append(line_p[k])
                        val_losses.append(lo)
                val_losses = np.mean(val_losses)
                val_acc = accuracy_score(ground_truth,predictions)
                val_f1 = f1_score(ground_truth,predictions,average='micro')
                print('val_losses:%f'%val_losses)
                print('val_accuracy:%f'%val_acc)
                print('val_f1:%f'%val_f1)
                s = sess.run(val_summary,feed_dict={val_losses_holder:val_losses,val_acc_holder:val_acc,
                                                      val_f1_holder:val_f1})
                writer.add_summary(s)




if __name__ == '__main__':
    config = Config()
    train(config)