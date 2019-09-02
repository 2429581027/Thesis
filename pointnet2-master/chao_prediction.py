#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 18:55:42 2019

@author: chaotang
"""

'''
    origin purpose:
        Evaluate classification performance with optional voting.
        Will use H5 dataset in default. If using normal, will shift to the normal dataset.
    New purpose: 
       try to use the trained model to make prediction     
'''
import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import modelnet_dataset
import modelnet_h5_dataset

import my_data_import as chao

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_cls_ssg', help='Model name. [default: pointnet2_cls_ssg]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores from multiple rotations [default: 1]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(ROOT_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))] 

HOSTNAME = socket.gethostname()

# Shapenet official train/test split
if FLAGS.normal:
    assert(NUM_POINT<=10000)
    DATA_PATH = os.path.join(ROOT_DIR, 'data/modelnet40_normal_resampled')
    TRAIN_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
    TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
else:
    assert(NUM_POINT<=2048)
    TRAIN_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=True)
    TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

# use this to predict the label for a specific mesh and output the feature vector
def prediction (num_votes):
    is_training = False
    with tf.device('/gpu:'+str(GPU_INDEX)):
       # pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        # def placeholder_inputs(batch_size, num_point):
        pointclouds_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
        labels_pl = tf.placeholder(tf.int32, shape=(BATCH_SIZE))

        # simple model
        # 尝试多返还一个变量给get model,这样就可以观察到此tensor 的值 
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
        
        MODEL.get_loss(pred, labels_pl, end_points)
        losses = tf.get_collection('losses')
        
        # add the current loss to the total loss
        total_loss = tf.add_n(losses, name='total_loss')
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")
    print('type pred: ',type(pred),'type l3_xyz : ',type(l3_xyz))
    # 输出结果显示这两个类型都是tensor
    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': total_loss,
           'l3_feature':l3_xyz} # 最后一行是自己加的 
    # eval_one_epoch(sess, ops, num_votes)
    
    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TEST_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    '''
    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        print('Batch: %03d, batch size: %d'%(batch_idx, bsize))
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        batch_pred_sum = np.zeros((BATCH_SIZE, NUM_CLASSES)) # score for classes
        for vote_idx in range(num_votes):
            # Shuffle point order to achieve different farthest samplings
            shuffled_indices = np.arange(NUM_POINT)
            np.random.shuffle(shuffled_indices)
            if FLAGS.normal:
                rotated_data = provider.rotate_point_cloud_by_angle_with_normal(cur_batch_data[:, shuffled_indices, :],
                    vote_idx/float(num_votes) * np.pi * 2)
            else:
                rotated_data = provider.rotate_point_cloud_by_angle(cur_batch_data[:, shuffled_indices, :],
                    vote_idx/float(num_votes) * np.pi * 2)
            feed_dict = {ops['pointclouds_pl']: rotated_data,
                         ops['labels_pl']: cur_batch_label,
                         ops['is_training_pl']: is_training}
            
            loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
            
            #试试看能不能计算这个tensor-----我草成了，能跑能跑！！
            # 下面两种形式都是可以跑的
            #loss_val, pred_val,check_l3 = sess.run([ops['loss'], ops['pred'],ops['l3_feature']], feed_dict=feed_dict)
            check_l3 = sess.run(l3_xyz,feed_dict = feed_dict)

            print('check_l3 shape= ',check_l3.shape)
            print('prediction = ', pred_val.shape)
            batch_pred_sum += pred_val
            
        pred_val = np.argmax(batch_pred_sum, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        batch_idx += 1
        # print('pred_val = ',pred_val.shape)
        # 一次会测试一个batch size 批次的mesh并给出预测结果.
        for i in range(bsize):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)
    
    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
     
    
    class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
    for i, name in enumerate(SHAPE_NAMES):
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))
    '''
    # my test
    # Shuffle point order to achieve different farthest samplings
    #
    # 该位置写上自己定义的读取想要几个飞机图像的函数
    cur_batch_data,cur_batch_label = chao.get_data()
    #

    shuffled_indices = np.arange(NUM_POINT)
    np.random.shuffle(shuffled_indices)
    rotated_data = provider.rotate_point_cloud_by_angle(cur_batch_data[:, shuffled_indices, :],
                    1/float(1) * np.pi * 2)
    feed_dict = {ops['pointclouds_pl']: rotated_data,
                         ops['labels_pl']: cur_batch_label,
                         ops['is_training_pl']: is_training}
    check_l3 = sess.run(l3_xyz,feed_dict = feed_dict)
    loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
    print('check_l3 shape= ',check_l3.shape)
    pred_val = np.argmax(pred_val, 1)
    
    my_correct = 0
    for i in range(4):
        l = cur_batch_label[i]
        total_seen_class[l] += 1
        my_correct += (pred_val[i] == l)
    accuracies = my_correct /total_seen_class[0]
    
    print("the classification accuracy is:" ,accuracies)
    print("total_seen_class = ",total_seen_class[0])
    print('total_correct_class',my_correct.shape)
    
    chao.draw_pointcloud(cur_batch_data[0,...])
    chao.draw_pointcloud(cur_batch_data[1,...])
    chao.draw_pointcloud(cur_batch_data[2,...])
    chao.draw_pointcloud(cur_batch_data[3,...])
    
    # check the feature map difference
    # between bed and plane
    diff1 =sum(sum((check_l3[0,...] - check_l3[3,...])**2))
    print('diff1 = ',diff1)
    # between plane
    diff2 =sum(sum((check_l3[0,...] - check_l3[1,...])**2))
    print('diff2 = ',diff2)
    
if __name__=='__main__':
    with tf.Graph().as_default():
        prediction(num_votes=FLAGS.num_votes)
    LOG_FOUT.close()
