# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 22:00:36 2020

@author: Administrator
"""
import numpy as np 
from scipy.spatial import cKDTree
import os
import shutil
import tensorflow as tf 
import os
import random
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--CUDA', type=int, default=0)
parser.add_argument('--dataset', type=str, default="shapenet")
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--class_idx', type=str, default="02691156")
a = parser.parse_args()

cuda_idx = str(a.CUDA)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cuda_idx

              

def safe_norm(x, epsilon=1e-12, axis=None):
    return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis) + epsilon)

def near_point_idx(array1, array2, num_point1,num_point2, num_features = 3):
    array1 = tf.reshape(array1,[-1,3])
    array2 = tf.reshape(array2,[-1,3])
    #num_point, num_features = array1.shape
    expanded_array1 = tf.tile(array1, (num_point2, 1))
    expanded_array2 = tf.reshape(
            tf.tile(tf.expand_dims(array2, 1), 
                    (1, num_point1, 1)),
            (-1, num_features))
    distances = safe_norm(expanded_array1-expanded_array2, axis=-1)
    distances = tf.reshape(distances, (num_point2, num_point1))
    dis_idx = tf.argmin(distances, axis=1)
    return dis_idx


points_target = tf.placeholder(tf.float32, shape=[1,None,3])
input_points_3d = tf.placeholder(tf.float32, shape=[1,None,3])
points_target_num = tf.placeholder(tf.int32, shape=[1,1])
points_input_num = tf.placeholder(tf.int32, shape=[1,1])
near_idx = near_point_idx(points_target,input_points_3d,points_target_num[0,0],points_input_num[0,0])
point_target_near = tf.gather(points_target,axis=1,indices=near_idx)

OUTPUT_DIR = a.out_dir
INPUT_DIR = a.input_dir

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    print ('test_res_dir: deleted and then created!')
os.makedirs(OUTPUT_DIR)


files = []

if(a.dataset=="shapenet"):
    f = open('./data/shapenet_val.txt','r')
    for index,line in enumerate(f):
        if(line.strip().split('/')[0]==a.class_idx):
            print(line)
            files.append(line.strip().split('/')[1])
    f.close()
    if len(files) == 0:
        with open(INPUT_DIR + 'val.lst') as f:
            data = f.readlines()
        files = [d.strip('\n') for d in data]
if(a.dataset == "famous"):
    f = open('./data/famous_testset.txt','r')
    for index,line in enumerate(f):
        #print(line)
        files.append(line.strip('\n'))
    f.close()
    
if(a.dataset == "ABC"):
    f = open('./data/abc_testset.txt','r')
    for index,line in enumerate(f):
        #print(line)
        files.append(line.strip('\n'))
    f.close()

if(a.dataset == "other"):
    fileAll = os.listdir(INPUT_DIR)
    for file in fileAll:
        if(re.findall(r'.*.xyz.npz', file, flags=0)):
            print(file.strip().split('.')[0])
            files.append(file.strip().split('.')[0])


# files = os.listdir(INPUT_DIR)
print('shap num:',len(files))
mm = 0
POINT_NUM = 300
POINT_NUM_GT = 300
POINT_NUM_GT_bs = np.array(POINT_NUM_GT).reshape(1,1)
points_input_num_bs = np.array(POINT_NUM).reshape(1,1)
with tf.Session() as sess:
    for file in files:
        print(file)

        print(OUTPUT_DIR + file + '.npz')
        if(a.dataset=="shapenet"):
            data = np.load(INPUT_DIR + file + '/pointcloud.npz')
            pointcloud = data['points'].reshape(-1,3)
            normal = data['normals'].reshape(-1,3)
        else:
            pointcloud = np.load(INPUT_DIR + file + '.xyz.npy').reshape(-1,3)
        
        point_idx = np.random.choice(pointcloud.shape[0], POINT_NUM_GT, replace = False)
        pointcloud = pointcloud[point_idx,:]
        print(np.max(pointcloud[:,0]),np.max(pointcloud[:,1]),np.max(pointcloud[:,2]),np.min(pointcloud[:,0]),np.min(pointcloud[:,1]),np.min(pointcloud[:,2]))
        pnts = pointcloud
        ptree = cKDTree(pnts)
        i = 0
        sigmas = []
        for p in np.array_split(pnts,100,axis=0):
            d = ptree.query(p,51)
            sigmas.append(d[0][:,-1])
            i = i+1
        
        sigmas = np.concatenate(sigmas)
        sigmas_big = 0.05 * np.ones_like(sigmas)
      
        sigmas = sigmas*2
        sample = []
        sample_near = []

        for i in range(200):
            tt = pnts + 1*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pnts.shape)
            sample.append(tt)
            tt = tt.reshape(-1,POINT_NUM,3)
            sample_t = []
            for j in range(tt.shape[0]):
                point_target_near_c = sess.run([point_target_near],feed_dict={points_target_num:POINT_NUM_GT_bs,points_input_num:points_input_num_bs,
                                               points_target:pnts.reshape(1,-1,3),input_points_3d:tt[j,:,:].reshape(1,-1,3)})
                point_target_near_c = np.asarray(point_target_near_c)
                point_target_near_c = point_target_near_c.reshape(-1,3)
                sample_t.append(point_target_near_c)
            sample_t = np.asarray(sample_t)
            sample_t = sample_t.reshape(-1,3)
            sample_near.append(sample_t)
            
        for i in range(100):
            if i < 100:
                tt = pnts + 0.5*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pnts.shape)
            else:
                tt = np.random.uniform(-0.5, 0.5, pnts.shape)
            sample.append(tt)
            tt = tt.reshape(-1,POINT_NUM,3)
            sample_t = []
            for j in range(tt.shape[0]):
                point_target_near_c = sess.run([point_target_near],feed_dict={points_target_num:POINT_NUM_GT_bs,points_input_num:points_input_num_bs,
                                               points_target:pnts.reshape(1,-1,3),input_points_3d:tt[j,:,:].reshape(1,-1,3)})
                point_target_near_c = np.asarray(point_target_near_c)
                point_target_near_c = point_target_near_c.reshape(-1,3)
                sample_t.append(point_target_near_c)
            sample_t = np.asarray(sample_t)
            sample_t = sample_t.reshape(-1,3)
            sample_near.append(sample_t)

        gt = pointcloud
        sample = np.asarray(sample)
        sample_near = np.asarray(sample_near)
        print('sample:',sample.shape,sample_near.shape)
 
        os.mkdir(OUTPUT_DIR + file)      

        if(a.dataset=="shapenet"): 
            np.savez(OUTPUT_DIR + file + '/' + file, sample = sample, point = pnts, normal = normal[point_idx,:],sample_near = sample_near, gt=gt)
        else:
            np.savez(OUTPUT_DIR + file + '/' + file, sample = sample, point = pnts, sample_near = sample_near, gt=gt)
