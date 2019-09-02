import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
import part_dataset_all_normal

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_part_seg', help='Model name [default: pointnet2_part_seg]')
parser.add_argument('--model_path', default='log_part_segmentation/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--log_dir', default='log_eval', help='Log dir [default: log_eval]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 32]')
FLAGS = parser.parse_args()

#VOTE_NUM = 12
# 稍微降低投票率可以提升运行速度
VOTE_NUM = 3


EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu

MODEL_PATH = FLAGS.model_path
MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
NUM_CLASSES = 50

# Shapenet official train/test split
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'shapenetcore_partanno_segmentation_benchmark_v0_normal')
TEST_DATASET = part_dataset_all_normal.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, 
                                                         classification=False, split='test')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print ('is_training_pl : ',is_training_pl)
            
            print ("--- Get model and loss")
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
            loss = MODEL.get_loss(pred, labels_pl)
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'feats':end_points['feats'],# 多加一个增加输出，这一层可以查看feature map
               'feats_0':end_points['feats_0']} # 这一层在全连接层之前的feature map，参数与上相同

        eval_one_epoch(sess, ops)

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 6))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    for i in range(bsize):
        ps,normal,seg = dataset[idxs[i+start_idx]]
        batch_data[i,:,0:3] = ps
        batch_data[i,:,3:6] = normal
        batch_label[i,:] = seg
    return batch_data, batch_label

def eval_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    # Test on all data: last batch might be smaller than BATCH_SIZE
    num_batches = (len(TEST_DATASET)+BATCH_SIZE-1)/BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    seg_classes = TEST_DATASET.seg_classes
    # seg_classes 包含以下形式
    #Airplane [0, 1, 2, 3]
    #Bag [4, 5]
    #Cap [6, 7]
    #Car [8, 9, 10, 11]
    #Chair [12, 13, 14, 15]
    #Earphone [16, 17, 18]
    #Guitar [19, 20, 21]
    #Knife [22, 23]
    #Lamp [24, 25, 26, 27]
    #Laptop [28, 29]
    #Motorbike [30, 31, 32, 33, 34, 35]
    #Mug [36, 37]
    #Pistol [38, 39, 40]
    #Rocket [41, 42, 43]
    #Skateboard [44, 45, 46]
    #Table [47, 48, 49]
    
    shape_ious = {cat:[] for cat in seg_classes.keys()}
     # shape_ious =  {'Earphone': [], 'Motorbike': [], 'Rocket': [], 
    #'Car': [], 'Laptop': [], 'Cap': [], 'Skateboard': [], 
    #'Mug': [], 'Guitar': [], 'Bag': [], 'Lamp': [], 'Table': [],
    #'Airplane': [], 'Pistol': [], 'Chair': [], 'Knife': []}  
    seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
    # 以下的代码会将seg_label_to_cat变成如下形式：
    # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    
    batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 6))
    batch_label = np.zeros((BATCH_SIZE, NUM_POINT)).astype(np.int32)
    # 加一个int在这里使得不会报错
    # 第一个batch全是飞机，飞机包含label0，1，2，3
    for batch_idx in range(1):#range(int(num_batches)):
        #if batch_idx %20==0:
        #   log_string('%03d/%03d'%(batch_idx, num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(len(TEST_DATASET), (batch_idx+1) * BATCH_SIZE)
        cur_batch_size = end_idx-start_idx
        cur_batch_data, cur_batch_label = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)
        
        if cur_batch_size == BATCH_SIZE:
            batch_data = cur_batch_data
            batch_label = cur_batch_label
        else:
            batch_data[0:cur_batch_size] = cur_batch_data
            batch_label[0:cur_batch_size] = cur_batch_label

        # ---------------------------------------------------------------------
        
        loss_val = 0
        pred_val = np.zeros((BATCH_SIZE, NUM_POINT, NUM_CLASSES))# (16*2048*50)
        feature_map = np.zeros((BATCH_SIZE,NUM_POINT,128)) # = (16*2048*128)
        for _ in range(VOTE_NUM):
            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['labels_pl']: batch_label,
                         ops['is_training_pl']: is_training}
            temp_loss_val, temp_pred_val,temp_feature_map = sess.run([ops['loss'], 
                                                                      ops['pred'],
                                                                      ops['feats']],feed_dict=feed_dict)
            loss_val += temp_loss_val
            pred_val += temp_pred_val
            feature_map += temp_feature_map
        print('pred_val = ',pred_val.shape)#np.argmax(pred_val, 2)[0,:])
        # print('temp_feature_map = ',temp_feature_map.shape)
        loss_val /= float(VOTE_NUM)
        feature_map /= float(VOTE_NUM)
        
        # ---------------------------------------------------------------------    
        # Select valid data
        cur_pred_val = pred_val[0:cur_batch_size]
        print('cur_pred_val = ', cur_pred_val.shape)
        # Constrain pred to the groundtruth classes (selected by seg_classes[cat])
        # 将预测结果限定在groundtruth classes
        cur_pred_val_logits = cur_pred_val
        cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
        # 以下部分进行了修改
        
        for i in range(cur_batch_size):
            cat = seg_label_to_cat[cur_batch_label[i,0]]
            logits = cur_pred_val_logits[i,:,:]
        # 输出的结果是可能性，以下代码的思路是只在正确cat里面去寻找概率高的，这样避免了整段错掉的情况
            #print('logits_origin = ',logits[:,seg_classes[cat]].shape)
            #print('logits = ',np.argmax(logits,axis=1))
            #print('logits_2 = ',np.argmax(logits[:,seg_classes[cat]], 1))
            #print('logits_3 = ',np.argmax(logits[:,seg_classes[cat]], 1) + seg_classes[cat][0])
            # 将预测结果限定在groundtruth classes
            
            #cur_pred_val[i,:] = np.argmax(logits[:,seg_classes[cat]], 1) + seg_classes[cat][0]
            
            # 为了方便plot，将代码该label的去掉
            
            cur_pred_val[i,:] = np.argmax(logits[:,seg_classes[cat]], 1)
            
            #print('seg_classes[cat]',seg_classes[cat])
            #print('seg_classes[cat][0]',seg_classes[cat][0])
        #print('cur_batch_label = ',cur_batch_label[0])
        #print('cur_pred_val = ',cur_pred_val[0])
        correct = np.sum(cur_pred_val[0] == cur_batch_label[0])
        total_correct += correct
        # 下面那个total_seen 会直接乘一个batch size，这个要注意
        total_seen += (cur_batch_size*NUM_POINT)
        
        if cur_batch_size==BATCH_SIZE:
            loss_sum += loss_val
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(cur_batch_label==l)
            total_correct_class[l] += (np.sum((cur_pred_val==l) & (cur_batch_label==l)))

        for i in range(cur_batch_size):
            segp = cur_pred_val[i,:]
            segl = cur_batch_label[i,:] 
            cat = seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
            for l in seg_classes[cat]:
                if (np.sum(segl==l) == 0) and (np.sum(segp==l) == 0): # part is not present, no prediction as well
                    part_ious[l-seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l-seg_classes[cat][0]] = np.sum((segl==l) & (segp==l)) / float(np.sum((segl==l) | (segp==l)))
            shape_ious[cat].append(np.mean(part_ious))
        #接下来的代码为plot点云图的代码
    # ---------------------------------------------------------------------
        colors = ['r','g','b','k','orange','magenta'] # 定义颜色，一个分类 最多 6 个分类因此最多6种颜色
        combine_data = np.zeros((BATCH_SIZE, NUM_POINT, 4))
        combine_data[:,:,0:3] = batch_data[:,:,0:3]
        
        combine_data[:,:,3] = cur_pred_val[:,:]
        # 将数据形式合并为x,y,z,label的形式
        # print('test data type',combine_data[0,combine_data[0,:,3]==0][:,0].shape)
        for which_batch in range(16):
            plt.figure()
            ax = plt.subplot(111,projection='3d')  # 创建一个三维的绘图工程
            for index in range(6):
                x = combine_data[which_batch,combine_data[which_batch,:,3]==index][:,0]
                y = combine_data[which_batch,combine_data[which_batch,:,3]==index][:,1]
                z = combine_data[which_batch,combine_data[which_batch,:,3]==index][:,2]
            
                ax.scatter(x,y,z,c = colors[index],s = 1)
    # 以下为代码原配的数据统计
    # ---------------------------------------------------------------------    
    all_shape_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_shape_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat])
    print ('all_shape_ious length = ',len(all_shape_ious))
    mean_shape_ious = np.mean(list(shape_ious.values()))
    log_string('eval mean loss: %f' % (loss_sum / float(len(TEST_DATASET)/BATCH_SIZE)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    for cat in sorted(shape_ious.keys()):
        log_string('eval mIoU of %s:\t %f'%(cat, shape_ious[cat]))
    log_string('eval mean mIoU: %f' % (mean_shape_ious))
    log_string('eval mean mIoU (all shapes): %f' % (np.mean(all_shape_ious)))
    
    
         
if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    evaluate()
    LOG_FOUT.close()
