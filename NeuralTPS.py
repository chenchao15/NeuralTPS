# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:44:22 2020

@author: Administrator
"""

import numpy as np
import tensorflow as tf 
import os 
import shutil
import random
import math
import scipy.io as sio
import time
from skimage import measure
# import binvox_rw
import argparse
import trimesh
from im2mesh.utils import libmcubes
from im2mesh.utils.libkdtree import KDTree
from p_encoder import get_2d_samples, patch_net
import re

parser = argparse.ArgumentParser()
parser.add_argument('--train',action='store_true', default=False)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--class_idx', type=str, default="026911156")
parser.add_argument('--save_idx', type=int, default=-1)
parser.add_argument('--CUDA', type=int, default=0)
parser.add_argument('--index', type=int, default=0)
parser.add_argument('--name', type=int, default=0)
parser.add_argument('--dataset', type=str, default="shapenet")
a = parser.parse_args()

cuda_idx = str(a.CUDA)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= cuda_idx


GT_NUM = 256
BS = 1
if GT_NUM == 500:
    POINT_NUM = 5000
else:   
    POINT_NUM = 2560 
POINT_NUM_GT = 20000
PATTERN_NUM = 1
list_path = a.data_dir[:-1] + '_list.txt'
with open(list_path, 'r') as f:
    name_list = f.readlines()
select_name = name_list[a.name].strip('\n')
INPUT_DIR = a.data_dir + select_name + '/'
INDEX = a.index 
OUTPUT_DIR = a.out_dir
if(a.dataset=='other'):
    GT_DIR = '/data/cc/data/ShapeNet/' + a.class_idx + '/'

TRAIN = a.train
bd = 0.55

if(TRAIN):
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print ('test_res_dir: deleted and then created!')
    os.makedirs(OUTPUT_DIR)
else:
    POINT_NUM = 128 * 128


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

#        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
#        # Handle normals that point into wrong direction gracefully
#        # (mostly due to mehtod not caring about this in generation)
#        normals_dot_product = np.abs(normals_dot_product)
        
        normals_dot_product = np.abs(normals_tgt[idx] * normals_src)
        normals_dot_product = normals_dot_product.sum(axis=-1)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def eval_pointcloud(pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None):
        ''' Evaluates a point cloud.

        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
        '''
        # Return maximum losses if pointcloud is empty


        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        completeness2 = completeness**2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        accuracy2 = accuracy**2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()
        #print(completeness,accuracy,completeness2,accuracy2)
        # Chamfer distance
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        print('chamferL2:',chamferL2)
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamferL1 = 0.5 * (completeness + accuracy)
        print('normals_correctness:',normals_correctness,'chamferL1:',chamferL1)
        return normals_correctness, chamferL1, chamferL2


def safe_norm_np(x, epsilon=1e-12, axis=1):
    return np.sqrt(np.sum(x*x, axis=axis) + epsilon)


def safe_norm(x, epsilon=1e-12, axis=None):
  return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis) + epsilon)


def gt_chamfer_distance(p, q):
    from nn_distance import tf_nndistance
    a,b,c,d = tf_nndistance.nn_distance(p, q)
    cd1 = tf.reduce_mean(a)
    cd2 = tf.reduce_mean(c)
    return cd1+cd2

def get_gt_and_weight(p, q, gt):
    from nn_distance import tf_nndistance
    a,b,c,d = tf_nndistance.nn_distance(p, q)
    gts = tf.gather(q[0], b)

    dis_for_w = get_distance(gts, gt)
    dis_for_w = tf.reduce_min(dis_for_w, 2)
    weight = dis_for_w
    weight = tf.exp(-10 * weight[:, :, None])
    return gts, weight


def get_distance(a, b):
    na = tf.reduce_sum(tf.square(a), 2)
    nb = tf.reduce_sum(tf.square(b), 2)
    na = tf.reshape(na, [BS, -1, 1])
    nb = tf.reshape(nb, [BS, 1, -1])
    dist = na - 2 * tf.matmul(a, b, False, True) + nb
    return dist


def get_tps_distance(a, b):
    na = tf.reduce_sum(tf.square(a), 2)
    nb = tf.reduce_sum(tf.square(b), 2)
    na = tf.reshape(na, [BS, -1, 1])
    nb = tf.reshape(nb, [BS, 1, -1])
    dist = na - 2 * tf.matmul(a, b, False, True) + nb
    dist = dist * tf.log(tf.abs(dist) + 1)
    return dist


def get_reg_loss(varss):
    reg_loss = tf.zeros([], dtype=tf.float32)
    for var in varss:
        if 'bias' not in varss:
            reg_loss += tf.nn.l2_loss(var)
    return reg_loss
    
    
files = []
files_path = []
if(a.dataset == "other"):
    fileAll = os.listdir(INPUT_DIR)
    for file in fileAll:
        if(re.findall(r'.*.npz', file, flags=0)):
            print(file.strip().split('.')[0])
            files.append(file.strip().split('.')[0])

for file in files:
    files_path.append(INPUT_DIR + file + '.npz')
SHAPE_NUM = len(files_path)
print('SHAPE_NUM:',SHAPE_NUM)

pointclouds = []
samples = []
base_gts = []
mm = 0
if(TRAIN):
    for file in files_path:
        load_data = np.load(file)
        base_gt = np.asarray(load_data['gt']).reshape([1, -1, 3])
        point = np.asarray(load_data['sample_near']).reshape(-1,POINT_NUM,3)
        sample = np.asarray(load_data['sample']).reshape(-1,POINT_NUM,3)
        pointclouds.append(point)
        samples.append(sample)
        base_gts.append(base_gt)
    pointclouds = np.asarray(pointclouds)
    samples = np.asarray(samples)
    base_gts = np.asarray(base_gts)
    print('data shape:',pointclouds.shape,samples.shape, base_gts.shape)
else:
    for file in files_path:
        load_data = np.load(file)
        base_gt = np.asarray(load_data['gt']).reshape([1, -1, 3])
        base_gts.append(base_gt)
    base_gts = np.asarray(base_gts)
    print('data shape:', base_gts.shape) 


feature = tf.placeholder(tf.float32, shape=[BS,None,SHAPE_NUM])
if TRAIN:
    input_points_3d = tf.placeholder(tf.float32, shape=[BS,POINT_NUM,3])
else:
    input_points_3d = tf.placeholder(tf.float32, shape=[BS,16384,3])
test_gt = tf.placeholder(tf.float32, shape=[BS,None,3])
test_generate = tf.placeholder(tf.float32, shape=[BS,None,3])
spase_gt_input = tf.placeholder(tf.float32, shape=[BS,None,3])
feature_f = tf.nn.relu(tf.layers.dense(feature,128))


def NeuralTPS(points_input, control_input):
    total_points = tf.concat([points_input, control_input], 1)
    net = tf.nn.relu(tf.layers.dense(total_points, 512))
    net = tf.concat([net, feature_f], 2)
    print('net:',net)
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        for i in range(8):
            with tf.variable_scope("resnetBlockFC_%d" % i ):
                b_initializer=tf.constant_initializer(0.0)
                w_initializer = tf.random_normal_initializer(mean=0.0,stddev=np.sqrt(2) / np.sqrt(512))
                net = tf.layers.dense(tf.nn.relu(net),512,kernel_initializer=w_initializer,bias_initializer=b_initializer)
    middle_net = tf.layers.dense(tf.nn.relu(net),128, kernel_initializer=w_initializer,bias_initializer=b_initializer)
    print('middle net:',middle_net)
    query_net = middle_net[:, :POINT_NUM]
    control_net = middle_net[:, POINT_NUM:]
    query_distance_net = get_tps_distance(query_net, control_net)
    control_distance_net = get_tps_distance(control_net, control_net)
    distance_net = tf.concat([query_distance_net, control_distance_net], 1)
    distance_net = tf.reshape(distance_net, [BS, -1, GT_NUM])
    b_initializer=tf.constant_initializer(-0.5)
    w_initializer = tf.random_normal_initializer(mean=2*np.sqrt(np.pi) / np.sqrt(512), stddev = 0.000001)
    sdf_fea = tf.layers.dense(tf.nn.relu(middle_net), 1, kernel_initializer=w_initializer, bias_initializer=b_initializer)
    sdf_tps = tf.layers.dense(tf.nn.relu(distance_net), 1, kernel_initializer=w_initializer, bias_initializer=b_initializer)
    sdf = sdf_fea + sdf_tps
    print('sdf',sdf)

    grad = tf.gradients(ys=sdf, xs=total_points) 
    print('grad',grad)
    normal_p_lenght = tf.expand_dims(safe_norm(grad[0],axis = -1),-1)
    print('normal_p_lenght',normal_p_lenght)
    grad_norm = grad[0]/normal_p_lenght
    print('grad_norm',grad_norm)

    surface_points = total_points - sdf * grad_norm

    return surface_points, sdf


def SurfaceParameterization(train_num, test_num):
    train_patch_num = int(train_num / PATTERN_NUM)  
    test_patch_num = int(test_num / PATTERN_NUM) 
    train_2d_samples = get_2d_samples(train_patch_num, PATTERN_NUM)
    train_2d_samples = train_2d_samples[:, None, :, :]
    test_2d_samples = get_2d_samples(test_patch_num, PATTERN_NUM)
    test_2d_samples = test_2d_samples[:, None, :, :] 
    is_training = tf.cast(True, 'bool')
    train_patches = []
    test_patches = []
    for i in range(PATTERN_NUM):
        patch_input = tf.concat([train_2d_samples[i], test_2d_samples[i]], 1)  
        total_patch = patch_net(patch_input, i, is_training, tf.nn.leaky_relu)
        train_patches.append(total_patch[:, :, :train_patch_num, :])
        test_patches.append(total_patch[:, :, train_patch_num:, :]) 
    train_patterns = tf.reshape(tf.concat(train_patches, 0), [BS, -1, 3])
    test_patterns = tf.reshape(tf.concat(test_patches, 0), [BS, -1, 3])
    return train_patterns, test_patterns


surface_points, sdf = NeuralTPS(input_points_3d, spase_gt_input)
surface_points = surface_points[:, :POINT_NUM]
control_sdf = sdf[:, POINT_NUM:]
sdf = sdf[:, :POINT_NUM]

sp_train, sp_test = SurfaceParameterization(2000, POINT_NUM)
sp_test = tf.stop_gradient(sp_test)
sp_test = tf.concat([sp_test, spase_gt_input], 1)

cd_loss = get_chamfer_distance(spase_gt_input, sp_train)

points_target, lpull_weight = get_gt_and_weight(input_points_3d, atlas_out_test, spase_gt_input)
lpull_loss = lpull_weight * tf.norm((points_target - surface_points), axis=-1)
lpull_loss = tf.reduce_mean(lpull_loss)

sdf_loss = tf.reduce_mean(control_sdf**2)

t_vars = tf.trainable_variables() 
reg_loss = get_reg_loss(t_vars)

loss = lpull_loss + 10 * cd_loss + sdf_loss + 1e-7 * reg_loss

optim = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9)
loss_grads_and_vars = optim.compute_gradients(loss, var_list=t_vars)
loss_optim = optim.apply_gradients(loss_grads_and_vars)

config = tf.ConfigProto(allow_soft_placement=False) 
saver_restore = tf.train.Saver(var_list=t_vars)
saver = tf.train.Saver(max_to_keep=2000000)

with tf.Session(config=config) as sess:
    feature_bs = []
    for i in range(SHAPE_NUM):
        tt = []
        for j in range(int(GT_NUM + POINT_NUM)):
            t = np.zeros(SHAPE_NUM)
            t[i] = 1
            tt.append(t)
        feature_bs.append(tt)
    feature_bs = np.asarray(feature_bs)

    if(TRAIN):
        print('train start')
        sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for i in range(40010):
            epoch_index = np.random.choice(SHAPE_NUM, SHAPE_NUM, replace = False)
            loss_i = 0
            for epoch in epoch_index:
                rt = random.randint(0,samples.shape[1]-1)
                input_points_2d_bs = samples[epoch,rt,:,:].reshape(BS, POINT_NUM, 3)
                feature_bs_t = feature_bs[epoch,:,:].reshape(1,-1,SHAPE_NUM)
                base_gt_i = base_gts[epoch]
                _, loss_c = sess.run([loss_optim, loss],feed_dict={input_points_3d:input_points_2d_bs,feature:feature_bs_t, spase_gt_input: base_gt_i})
                loss_i = loss_i + loss_c
            loss_i = loss_i / SHAPE_NUM
            if(i%10 == 0):
                print('epoch:', i, 'epoch loss:', loss_i)
            if(i%5000 == 0):
                print('save model')
                saver.save(sess, os.path.join(OUTPUT_DIR, "model"), global_step=i+1)
        end_time = time.time()
        print('run_time:',end_time-start_time)
    else:
        print('test start')
        checkpoint = tf.train.get_checkpoint_state(OUTPUT_DIR).all_model_checkpoint_paths
        path = OUTPUT_DIR + 'model-' + str(INDEX * 5000 + 1)
        print(path)
        saver.restore(sess, path)
        s = np.arange(-bd,bd, (2*bd)/128)
            
        print(s.shape[0])
        vox_size = s.shape[0]
        input_points_2d_bs = []
        for i in s:
            for j in s:
                for k in s:
                    input_points_2d_bs.append(np.asarray([i,j,k]))
        input_points_2d_bs = np.asarray(input_points_2d_bs)
        print('input_points_2d_bs',input_points_2d_bs.shape)
        
        test_num = SHAPE_NUM
        print('test_num:',test_num)
        cd = 0
        nc = 0
        cd2 = 0
        for epoch in range(test_num):
            print('test:',epoch)
            vox = []
            feature_bs = []
            for j in range(GT_NUM + POINT_NUM):
                t = np.zeros(SHAPE_NUM)
                t[epoch] = 1
                feature_bs.append(t)
            feature_bs = np.asarray(feature_bs)
            
            totaln = int(vox_size * vox_size * vox_size / POINT_NUM)
            if vox_size * vox_size * vox_size % POINT_NUM:
                totaln += 1
         
            for i in range(totaln):
                input_points_2d_bs_t = input_points_2d_bs[i * POINT_NUM : (i+1)*POINT_NUM,:]
                input_points_2d_bs_t = input_points_2d_bs_t.reshape(BS, POINT_NUM, 3)
                feature_bs_t = feature_bs.reshape(BS,GT_NUM + POINT_NUM,SHAPE_NUM)
                base_gt_i = base_gts[epoch]
                sdf_c = sess.run([sdf], feed_dict={input_points_3d:input_points_2d_bs_t, feature:feature_bs_t, spase_gt_input: base_gt_i})
                vox.append(sdf_c)
                
            vox = np.asarray(vox)
            vox = vox.reshape((vox_size,vox_size,vox_size))
            vox_max = np.max(vox.reshape((-1)))
            vox_min = np.min(vox.reshape((-1)))
            print('max_min:',vox_max,vox_min)
            
            threshs = [0.005]
            for thresh in threshs:
                print(np.sum(vox>thresh),np.sum(vox<thresh))
                
                if(np.sum(vox>0.0)<np.sum(vox<0.0)):
                    thresh = -thresh
                print('model:',epoch,'thresh:',thresh)
                vertices, triangles = libmcubes.marching_cubes(vox, thresh)
                if(vertices.shape[0]<10 or triangles.shape[0]<10):
                    print('no surface')
                    continue
                if(np.sum(vox>0.0)>np.sum(vox<0.0)):
                    triangles_t = []
                    for it in range(triangles.shape[0]):
                        tt = np.array([triangles[it,2],triangles[it,1],triangles[it,0]])
                        triangles_t.append(tt)
                    triangles_t = np.asarray(triangles_t)
                else:
                    triangles_t = triangles
                    triangles_t = np.asarray(triangles_t)

                vertices -= 0.5
                # Undo padding
                vertices -= 1
                # Normalize to bounding box
                vertices /= np.array([vox_size-1, vox_size-1, vox_size-1])
                vertices = 1.1 * (vertices - 0.5)
                mesh = trimesh.Trimesh(vertices, triangles_t,
                               vertex_normals=None,
                               process=False)
                mesh.export(OUTPUT_DIR +  '/neuraltps_' + files[epoch] + '_'+ str(INDEX*100 + 1) +  str(threshs[0]) + '.off')
                
    
                mesh = trimesh.Trimesh(vertices, triangles,
                                   vertex_normals=None,
                                   process=False)
                                   
                if(a.dataset=="shapenet" or a.dataset=='other'):
                    ps, idx = mesh.sample(1000000, return_index=True)
                else:
                    ps, idx = mesh.sample(10000, return_index=True)
                ps = ps.astype(np.float32)
                normals_pred = mesh.face_normals[idx]
                
                if (a.dataset=="shapenet" or a.dataset == 'other'):
                    data = np.load(GT_DIR + files[epoch] + '/pointcloud.npz')
                    pointcloud = data['points']
                    normal = data['normals']
                else:
                    mesh_gt = trimesh.load(GT_DIR + files[epoch] + '.ply')
                    pointcloud = pointcloud.astype(np.float32)
                    normal = mesh_gt.face_normals[idx_gt]
                
                nc_t,cd_t,cd2_t = eval_pointcloud(ps,pointcloud.astype(np.float32),normals_pred.astype(np.float32),normal.astype(np.float32))
                np.savez(OUTPUT_DIR + files[epoch]+ '_'+ str(thresh),pp = ps, np = normals_pred, p = pointcloud, n = normal, nc = nc_t, cd = cd_t, cd2 = cd2_t)
                nc = nc + nc_t
                cd = cd + cd_t
                cd2 = cd2 + cd2_t
        with open(OUTPUT_DIR +  '/nc_cd2_' + files[epoch] + '_'+ str(INDEX*100 + 1) +  str(threshs[0]) + '.txt', 'w') as f:
            f.write(str(nc/test_num))
            f.write(str(cd2/test_num))

        print('mean_nc:',nc/test_num,'mean_cd:',cd/test_num,'cd2:',cd2/test_num)
                    
    

    
