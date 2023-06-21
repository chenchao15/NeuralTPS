#coding=utf-8
import numpy as np
import tensorflow as tf
import tf_util

slim = tf.contrib.slim


def _preprocess(images):
    return images * 2 - 1

def patch_net(input, index, is_training, act_func):
    pattern = input[None, :, :, :]
    m = [128, 256]
    for i in range(len(m)):
        pattern = tf_util.conv2d(pattern, m[i], [1,3],
                   padding='SAME', stride=[1,1],  
                   bn=True, is_training=is_training,
                   scope='init_pattern_0_' + str(i) + '_' + str(index), bn_decay=None)
    m = [512, 256]
    for i in range(len(m)):
        pattern = tf_util.conv2d(pattern, m[i], [1, 1],
                   padding='SAME', stride=[1,1], 
                   bn=True, is_training=is_training, 
                   scope='init_pattern_1_' + str(i) + '_' + str(index), bn_decay=None, activation_fn=act_func)
    pattern = tf_util.conv2d(pattern, 3, [1, 1],
                   padding='SAME', stride=[1,1],
                   bn=True, is_training=is_training, 
                   scope='init_pattern_2_' + str(index), bn_decay=None, activation_fn=tf.tanh)
    
    return pattern
    

def get_2d_samples(pointnum, n):
    samples = tf.random_uniform([n, pointnum, 2], 0.0, 1.0)
    return samples
