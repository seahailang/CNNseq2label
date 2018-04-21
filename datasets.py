#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: datasets.py.py
@time: 2018/4/18 16:32
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import utils

FLAGS = tf.app.flags.FLAGS

def make_train_data(seq,label,seq_len):
    datasets1 = tf.data.Dataset.from_tensor_slices(seq)
    datasets2 = tf.data.Dataset.from_tensor_slices(label)
    datasets3 = tf.data.Dataset.from_tensor_slices(seq_len)

    dataset = tf.data.Dataset.zip([datasets1,datasets2,datasets3])
    return dataset

if __name__ == '__main__':
    pass