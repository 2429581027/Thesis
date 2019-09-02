#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:37:50 2019

@author: chaotang
"""
import os,sys
BASE_DIR = os.getcwd()
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from pointnet_util import pointnet_sa_module

# 剩下的函数全部写在这里

def get_model(point_cloud,is_training,bn_decay = None):
    #  定义BN层的参数
    bn_decay = bn_decay if bn_decay is not None else 0.9
    #
    batch_size = point_cloud.shape[0]
    num_point = point_cloud.shape[1]
    end_points = {}
    l0_xyz = point_cloud
    l0_points = None
    end_points['l0_xyz'] = l0_xyz
    # Set abstraction layers
    # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
    # So we only use NCHW for layer 1 until this issue can be resolved.
    ## First Half Net
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(
            l0_xyz, l0_points, npoint=512, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None,
            group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1', use_nchw=True)
    
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(
            l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, 
            group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(
            l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None,
            group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    ### Second Half Net(New)
    Flat = Flatten()(l3_points)
    fc1 = Dense(512,activation = 'relu',kernel_initializer = 'he_uniform',name = 'FC1')(Flat)
    fcbn1 = tf.contrib.layers.batch_norm(fc1, center=True, scale=True,is_training=is_training, 
                                         decay=bn_decay,updates_collections=None,scope='fcbn1')
    fc2 = Dense(256,activation = 'relu',kernel_initializer = 'he_uniform',name = 'FC2')(fcbn1)
    fcbn2 = tf.contrib.layers.batch_norm(fc2, center=True, scale=True,is_training=is_training, 
                                         decay=bn_decay,updates_collections=None,scope='fcbn2')
    pred = Dense(40, activation = 'relu',kernel_initializer = 'he_uniform', name='FC3')(fcbn2)
      
    return pred,end_points

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        output, _ = get_model(inputs, tf.constant(True))
        print(output)