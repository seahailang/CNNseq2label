#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: utils.py
@time: 2018/4/18 16:32
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import re
import numpy as np
import spacy

FLAGS = tf.app.flags.FLAGS

def preprocess(data_dir ='./train2'):
    filenames = os.listdir(data_dir)
    filenames = map(lambda x:x.split('.')[0],filenames)
    filenames = set(filenames)
    datas = []
    labels = []
    vocabulary = set([])
    labels_vocab = set([])
    for filename in filenames:
        text_name = os.path.join(data_dir,filename+'.txt')
        ann_name = os.path.join(data_dir,filename+'.ann')
        with open(text_name,encoding='utf-8') as file:
            text = file.read().strip()
        with open(ann_name,encoding='utf-8') as file:
            ann = []
            for line in file.readlines():
                if line.startswith('T'):
                    l = line.split('\t')[1]
                    temp = l.split(' ')
                    label=temp[0]
                    if label=='T':
                        print('T')
                    labels_vocab.add(label)
                    x = temp[1]
                    y=temp[-1]
                    ann.append((label,int(x),int(y)))
        i = 0
        j = 0
        text2ann = []
        for item in text.split(' '):
            token = re.findall(r'[^(.,?)}]+',item.lower())[0]
            vocabulary.add(token)
            if j<len(ann):
                if i >= ann[j][1] and i < ann[j][2]:
                    text2ann.append((token,ann[j][0]))
                else:
                    text2ann.append((token,'C'))
                i += len(item)+1
                    # text2ann.append((item,'XXX'))
                if i >= ann[j][2]:
                    j += 1
        datas.append(np.array(text2ann)[:,0])
        labels.append(np.array(text2ann)[:,1])
    return datas,labels,vocabulary,labels_vocab

def load_vector(vocabulary,vector_dir = './glove.6B.200d.txt',dims=200):
    word_ids= {}
    for vocab in vocabulary:
        word_ids[vocab] = 0
    matrixs = [np.zeros((dims,))]
    with open(vector_dir,'r',encoding='utf-8') as file:
        i = 1
        for line in file.readlines():
            word = line.split(' ')[0]
            arr = np.array(line.split(' ')[1:]).astype(np.float32)
            if word in word_ids:
                word_ids[word] = i
                matrixs.append(arr)
                i+=1
    return word_ids,np.array(matrixs)

def texts2seqs(datas,word_ids,max_len=300):
    new_data = []
    seq_len = []
    for data in datas:
        data = list(map(lambda x:word_ids[x],data))
        if len(data)<=max_len:
            data += [0]*(max_len-len(data))
            seq_len.append(len(data))
        else:
            data = data[:max_len]
            seq_len.append(max_len)

        new_data.append(data)
    return np.array(new_data),np.array(seq_len)

def load_data(config):
    if os.path.exists('./data.npz'):
        npz = np.load('./data.npz')
        seq = npz['sequence']
        seq_label = npz['label']
        seq_len = npz['length']
        matrixs = npz['matrix']
    else:
        datas, labels, vocabulary, l_vocabulary = preprocess(config.data_dir)
        l_vocabulary.add('C')
        word_ids, matrixs = load_vector(vocabulary)
        label_ids = {}
        for i, label in enumerate(sorted(l_vocabulary)):
            label_ids[label] = i
        seq, seq_len = texts2seqs(datas=datas, word_ids=word_ids, max_len=config.max_len)
        seq_label, _ = texts2seqs(datas=labels, word_ids=label_ids, max_len=config.max_len)
        np.savez('./data.npz',sequence = seq,label = seq_label,length = seq_len,matrix = matrixs)
    return seq,seq_label,seq_len,matrixs



def dot_attention(A,B):
    B_T = tf.transpose(B,[0,2,1])
    sim = tf.nn.softmax(tf.matmul(A,B_T))
    new_A = tf.matmul(sim,B)
    return new_A

def conject_dot_attention(A,B,W):
    _W = tf.matmul(W,tf.transpose(W,[1,0]))
    B_T = tf.transpose(B,[0,2,1])
    sim = tf.nn.softmax(A@_W@B_T)
    new_A = sim@B
    return new_A




# def pre(data_dir ='./train2'):
#     nlp = spacy.load('en')
#     filenames = os.listdir(data_dir)
#     filenames = map(lambda x:x.split('.')[0],filenames)
#     filenames = set(filenames)
#     datas = []
#     labels = []
#     vocabulary = set([])
#     for filename in filenames:
#         text_name = os.path.join(data_dir,filename+'.txt')
#         ann_name = os.path.join(data_dir,filename+'.ann')
#         docs = nlp(text_name)










if __name__ == '__main__':
    datas,labels,vocabulary,l_vocabulary = preprocess()
    l_vocabulary.add('C')
    word_ids,matrixs = load_vector(vocabulary)
    label_ids = {}
    for i, label in enumerate(sorted(l_vocabulary)):
        label_ids[label] =i
    seq,seq_len = texts2seqs(datas=datas,word_ids=word_ids,max_len=500)
    seq_label,_ = texts2seqs(datas=labels,word_ids=label_ids,max_len=500)
    i = 0

