#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:26:10 2019

@author: chaotang
read .txt file and covert it into a numpy array.
this is for the 
"""
import numpy as np
import modelnet_dataset
import os
import sys

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
DATA_PATH = os.path.join(ROOT_DIR, 'data/modelnet40_normal_resampled')
NUM_POINT = 2048
BATCH_SIZE = 16

def pc_normalize(pc):
    # l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def get_data():
    # initialization
    batch_data = np.zeros((BATCH_SIZE,NUM_POINT,3))
    batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)
    # the First pcloud
    point_set = np.loadtxt(DATA_PATH + "/laptop/laptop_0077.txt",delimiter=',').astype(np.float32)
    point_set = point_set[0:NUM_POINT,0:3]

    batch_data[0,...] = pc_normalize(point_set)
    batch_label[0] = 20

    # the second point cloud
    point_set2 = np.loadtxt(DATA_PATH + "/laptop/laptop_0002.txt",delimiter=',').astype(np.float32)
    point_set2 = point_set2[0:NUM_POINT,0:3]
    
    batch_data[1,...] = pc_normalize(point_set2)
    batch_label[1] = 20
  
    # the thrid point cloud
    point_set3 = np.loadtxt(DATA_PATH + "/laptop/laptop_0003.txt",delimiter=',').astype(np.float32)
    point_set3 = point_set3[0:NUM_POINT,0:3]
    
    batch_data[2,...] = pc_normalize(point_set3)
    batch_label[2] = 20
    
    # the fouth point cloud
    point_set4 = np.loadtxt(DATA_PATH + "/laptop/laptop_0004.txt",delimiter=',').astype(np.float32)
    point_set4 = point_set4[0:NUM_POINT,0:3]
    
    batch_data[3,...] = pc_normalize(point_set4)
    batch_label[3] = 20
    
    for i in range(1,10):
        fileName = DATA_PATH + "/laptop/laptop_000" + str(i) + ".txt"
        #print(fileName)
        point_set = np.loadtxt(fileName,delimiter=',').astype(np.float32)
        point_set = point_set[0:NUM_POINT,0:3]

        batch_data[i,...] = pc_normalize(point_set)
        batch_label[i] = 20
        
    for i in range(10,16):
        fileName = DATA_PATH + "/laptop/laptop_00" + str(i) + ".txt"
        #print(fileName)
        point_set = np.loadtxt(fileName,delimiter=',').astype(np.float32)
        point_set = point_set[0:NUM_POINT,0:3]

        batch_data[i,...] = pc_normalize(point_set)
        batch_label[i] = 20
        
    
    return batch_data,batch_label

def draw_pointcloud(batch_data):
    plt.figure()
    x,y,z = batch_data[:,0],batch_data[:,1],batch_data[:,2]
    ax = plt.subplot(111, projection='3d' )  # 创建一个三维的绘图工程
    ax.scatter(x,y,z,c = 'r',s = 1)
    

if __name__=='__main__':
    BATCH_SIZE = 16
    NUM_POINT = 1024
    batch_data = np.zeros((BATCH_SIZE,NUM_POINT,3))
    batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)
    batch_data,batch_label = get_data()
    print(batch_data.shape)
    draw_pointcloud(batch_data[0])
