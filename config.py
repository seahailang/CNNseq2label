#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: config.py
@time: 2018/4/18 16:32
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class Config(object):
    def __init__(self):
        self.root_dir = './'
        self.max_train_steps = 100000
        self.val_steps = 100
        self.ckpt = self.root_dir+'ckpt/'
        self.data_dir = self.root_dir+'train2/'
        self.optimizer = 'adam'
        self.max_len = 2000
        self.batch_size = 8
        self.learning_rate = 0.001
        self.mode = 'train'

FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    pass