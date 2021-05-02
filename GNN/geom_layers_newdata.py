import tensorflow.compat.v1 as tf
import numpy as np
from utils import glorot, zeros, block_diagonal, preprocess_adj
flags = tf.app.flags
FLAGS = flags.FLAGS
_LAYER_UIDS = {}
import shellconv

def get_layer_uid(layer_name=''):
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def batch_distance_matrix(A):
    r = tf.reduce_sum(A * A, axis=2, keepdims=True)
    m = tf.matmul(A, tf.transpose(A, perm=(0, 2, 1)))
    D = r - 2 * m + tf.transpose(r, perm=(0, 2, 1))
    return D

def sparse_dropout(x, keep_prob, noise_shape):
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def dot(x, y, sparse=False):
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

class Layer(object):
    def __init__(self, **kwargs):
        layer = self.__class__.__name__.lower()
        name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.weights = {}
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs
            
class ConvolutionalLayer(Layer):
    def __init__(self, input_dim, output_dim, placeholders, dropout,
                 sparse_inputs, activation,   bias=False,  **kwargs):
        super(ConvolutionalLayer, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        self.output_dim = output_dim
        self.adj = placeholders['adj']
        self.dist_adj = placeholders['dist_adj']
        self.activation = activation
        self.sparse_inputs = sparse_inputs
        self.bias = bias
        #modify here
        self.bias_mat = placeholders['bias_mat']
        self.coords = placeholders['coords']
        
        with tf.variable_scope(self.name + '_weights'):
            
            self.weights['weights'] = glorot([input_dim, output_dim],
                                                        name='weights')
            if self.bias:
                self.weights['bias'] = zeros([output_dim], name='bias')

    def _call(self, inputs):
        x = inputs

        # dropout
        
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)
        
        # convolve
        
        pre_sup = dot(x, self.weights['weights'],
                      sparse=self.sparse_inputs)
        
        
        
        adjs = [tf.nn.softmax(self.adj + self.bias_mat[i]) for i in range(len(self.bias_mat))] 
        dist_adjs = [tf.nn.softmax(- self.dist_adj + self.bias_mat[i]) for i in range(len(self.bias_mat))] 
        support = adjs + dist_adjs
        #注意这里要有一个负号，才能使得距离越小权重越大
        
        support = [preprocess_adj(adj, False, False) for adj in support]
        support = [dot(support[i], pre_sup, sparse=False) for i in range(len(support))]
     
        #convolution
        support = tf.concat(tuple(support),axis=-1)  
        # this line should be paid attention when change the number of geometric relations
        #support = tf.reshape(support, [FLAGS.batchsize, num_nodes, 8, -1])
        #output = tf.layers.conv2d(support, self.output_dim*8, kernel_size=[1,8], strides=(1, 8), padding='VALID',activation=self.activation)
        #output = tf.reshape(output, [FLAGS.batchsize*num_nodes, -1])
        output = support 
        return output
        
        # bias
        '''
        output = tf.concat(tuple(support),axis=-1) 
        if self.bias:
            output += self.weights['bias']

        return self.activation(output)
        '''

class DenseLayer(Layer):
    def __init__(self,output_dim,activation, placeholders, dropout):
        super(DenseLayer,self).__init__()
        self.output_dim = output_dim
        self.activation = activation
        self.placeholders = placeholders
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        return tf.layers.dense(inputs, self.output_dim, activation=self.activation, name=self.name)
        
class PoolingLayer(Layer):
    def __init__(self, placeholders,
                 activation,  bias=False,  **kwargs):
        super(PoolingLayer, self).__init__(**kwargs)

        
        self.activation = activation
        self.pooling_matrix = placeholders['pooling_matrix']
    
    def _call(self, inputs):
        
        output = dot(self.pooling_matrix, inputs, sparse = False)
        return self.activation(output)
class ShellConvLayer():
    def __init__(self, is_training, tag, K, D, P, C, with_local, bn_decay=None):
        layer = self.__class__.__name__.lower()
        name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.weights = {}
        self.K = K
        self.D = D
        self.P = P
        self.C = C
        self.is_training = is_training
        self.tag = tag
        self.with_local = with_local
        self.bn_decay =  bn_decay
    def __call__(self, points, features_prev, queries):
        with tf.name_scope(self.name):
            outputs = shellconv.shellconv(points, features_prev, queries, self.is_training, self.tag, self.K, self.D, self.P, self.C, self.with_local, self.bn_decay)
            return outputs