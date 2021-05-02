import math
import random
import itertools
import numpy as np
import tensorflow.compat.v1 as tf
from numpy import linalg as LA

# the returned indices will be used by tf.gather_nd
def get_indices(batch_size, sample_num, point_num, pool_setting=None):
    if not isinstance(point_num, np.ndarray):
        point_nums = np.full((batch_size), point_num)
    else:
        point_nums = point_num

    indices = []
    for i in range(batch_size):
        pt_num = point_nums[i]
        if pool_setting is None:
            pool_size = pt_num
        else:
            if isinstance(pool_setting, int):
                pool_size = min(pool_setting, pt_num)
            elif isinstance(pool_setting, tuple):
                pool_size = min(random.randrange(pool_setting[0], pool_setting[1]+1), pt_num)
        if pool_size > sample_num:
            choices = np.random.choice(pool_size, sample_num, replace=False)
        else:
            choices = np.concatenate((np.random.choice(pool_size, pool_size, replace=False),
                                      np.random.choice(pool_size, sample_num - pool_size, replace=True)))
        if pool_size < pt_num:
            choices_pool = np.random.choice(pt_num, pool_size, replace=False)
            choices = choices_pool[choices]
        choices = np.expand_dims(choices, axis=1)
        choices_2d = np.concatenate((np.full_like(choices, i), choices), axis=1)
        indices.append(choices_2d)
    return np.stack(indices)

def gauss_clip(mu, sigma, clip):
    v = random.gauss(mu, sigma)
    v = max(min(v, mu + clip * sigma), mu - clip * sigma)
    return v

def uniform(bound):
    return bound * (2 * random.random() - 1)


def scaling_factor(scaling_param, method):
    try:
        scaling_list = list(scaling_param)
        return random.choice(scaling_list)
    except:
        if method == 'g':
            return gauss_clip(1.0, scaling_param, 3)
        elif method == 'u':
            return 1.0 + uniform(scaling_param)


def rotation_angle(rotation_param, method):
    try:
        rotation_list = list(rotation_param)
        return random.choice(rotation_list)
    except:
        if method == 'g':
            return gauss_clip(0.0, rotation_param, 3)
        elif method == 'u':
            return uniform(rotation_param)


def get_xforms(xform_num, rotation_range=(0, 0, 0, 'u'), scaling_range=(0.0, 0.0, 0.0, 'u'), order='rxyz'):
    xforms = np.empty(shape=(xform_num, 3, 3))
    rotations = np.empty(shape=(xform_num, 3, 3))
    for i in range(xform_num):
        rx = rotation_angle(rotation_range[0], rotation_range[3])
        ry = rotation_angle(rotation_range[1], rotation_range[3])
        rz = rotation_angle(rotation_range[2], rotation_range[3])
        rotation = euler2mat(rx, ry, rz, order)

        sx = scaling_factor(scaling_range[0], scaling_range[3])
        sy = scaling_factor(scaling_range[1], scaling_range[3])
        sz = scaling_factor(scaling_range[2], scaling_range[3])
        scaling = np.diag([sx, sy, sz])

        xforms[i, :] = scaling * rotation
        rotations[i, :] = rotation
    return xforms, rotations



def augment(points, xforms, range=None):
    points_xformed = tf.matmul(points, xforms, name='points_xformed')
    if range is None:
        return points_xformed

    jitter_data = range * tf.random_normal(tf.shape(points_xformed), name='jitter_data')
    jitter_clipped = tf.clip_by_value(jitter_data, -5 * range, 5 * range, name='jitter_clipped')
    return points_xformed + jitter_clipped


# A shape is (N, C)
def distance_matrix(A):
    r = tf.reduce_sum(A * A, 1, keepdims=True)
    m = tf.matmul(A, tf.transpose(A))
    D = r - 2 * m + tf.transpose(r)
    return D


# A shape is (N, P, C)
def batch_distance_matrix(A):
    r = tf.reduce_sum(A * A, axis=2, keepdims=True)
    m = tf.matmul(A, tf.transpose(A, perm=(0, 2, 1)))
    D = r - 2 * m + tf.transpose(r, perm=(0, 2, 1))
    return D


# A shape is (N, P_A, C), B shape is (N, P_B, C)
# D shape is (N, P_A, P_B)
def batch_distance_matrix_general(A, B):
    r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
    r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
    m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
    D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
    return D


# A shape is (N, P, C)
def find_duplicate_columns(A):
    N = A.shape[0]
    P = A.shape[1]
    indices_duplicated = np.fill((N, 1, P), 1, dtype=np.int32)
    for idx in range(N):
        _, indices = np.unique(A[idx], return_index=True, axis=0)
        indices_duplicated[idx, :, indices] = 0
    return indices_duplicated


# add a big value to duplicate columns
def prepare_for_unique_top_k(D, A):
    #indices_duplicated = tf.py_function(find_duplicate_columns, [A], tf.int32)
    indices_duplicated = tf.py_func(find_duplicate_columns, [A], tf.int32)
    D += tf.reduce_max(D)*tf.cast(indices_duplicated, tf.float32)


# return shape is (N, P, K, 2)
def knn_indices(points, k, sort=True, unique=True):
    points_shape = tf.shape(points)
    batch_size = points_shape[0]
    point_num = points_shape[1]

    D = batch_distance_matrix(points)
    if unique:
        prepare_for_unique_top_k(D, points)
    distances, point_indices = tf.nn.top_k(-D, k=k, sorted=sort)
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, point_num, k, 1))
    indices = tf.concat([batch_indices, tf.expand_dims(point_indices, axis=3)], axis=3)
    return -distances, indices


# return shape is (N, P, K, 2)
def knn_indices_general(queries, points, k, sort=True, unique=True):
    queries_shape = tf.shape(queries)
    batch_size = queries_shape[0]
    point_num = queries_shape[1]
    tmp_k = 0
    D = batch_distance_matrix_general(queries, points)
    if unique:
        prepare_for_unique_top_k(D, points)
    _, point_indices = tf.nn.top_k(-D, k=k+tmp_k, sorted=sort)  # (N, P, K)
    # point_indices = tf.contrib.framework.argsort(D)
    # point_indices = point_indices[:,:,:k]
    
    #batch_indices[i,:,:,:]的值都是i   
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, point_num, k, 1)) #(N,P,K,1)
    indices = tf.concat([batch_indices, tf.expand_dims(point_indices[:,:,tmp_k:], axis=3)], axis=3)
    return indices


# indices is (N, P, K, 2)
# return shape is (N, P, K, 2)
def sort_points(points, indices, sorting_method):
    indices_shape = tf.shape(indices)
    batch_size = indices_shape[0]
    point_num = indices_shape[1]
    k = indices_shape[2]

    nn_pts = tf.gather_nd(points, indices)  # (N, P, K, 3)
    if sorting_method.startswith('c'):
        if ''.join(sorted(sorting_method[1:])) != 'xyz':
            print('Unknown sorting method!')
            exit()
        epsilon = 1e-8
        nn_pts_min = tf.reduce_min(nn_pts, axis=2, keepdims=True)
        nn_pts_max = tf.reduce_max(nn_pts, axis=2, keepdims=True)
        nn_pts_normalized = (nn_pts - nn_pts_min) / (nn_pts_max - nn_pts_min + epsilon)  # (N, P, K, 3)
        scaling_factors = [math.pow(100.0, 3 - sorting_method.find('x')),
                           math.pow(100.0, 3 - sorting_method.find('y')),
                           math.pow(100.0, 3 - sorting_method.find('z'))]
        scaling = tf.constant(scaling_factors, shape=(1, 1, 1, 3))
        sorting_data = tf.reduce_sum(nn_pts_normalized * scaling, axis=-1)  # (N, P, K)
        sorting_data = tf.concat([tf.zeros((batch_size, point_num, 1)), sorting_data[:, :, 1:]], axis=-1)
    elif sorting_method == 'l2':
        nn_pts_center = tf.reduce_mean(nn_pts, axis=2, keepdims=True)  # (N, P, 1, 3)
        nn_pts_local = tf.subtract(nn_pts, nn_pts_center)  # (N, P, K, 3)
        sorting_data = tf.norm(nn_pts_local, axis=-1)  # (N, P, K)
    else:
        print('Unknown sorting method!')
        exit()
    _, k_indices = tf.nn.top_k(sorting_data, k=k, sorted=True)  # (N, P, K)
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, point_num, k, 1))
    point_indices = tf.tile(tf.reshape(tf.range(point_num), (1, -1, 1, 1)), (batch_size, 1, k, 1))
    k_indices_4d = tf.expand_dims(k_indices, axis=3)
    sorting_indices = tf.concat([batch_indices, point_indices, k_indices_4d], axis=3)  # (N, P, K, 3)
    return tf.gather_nd(indices, sorting_indices)


