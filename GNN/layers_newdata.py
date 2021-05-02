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
                 sparse_inputs, activation,  bias=False,  **kwargs):
        super(ConvolutionalLayer, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        self.adj = placeholders['adj']
        self.activation = activation
        self.sparse_inputs = sparse_inputs
        self.bias = bias
        
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
        
        
        # construct support(normalized laplcian)
        support = self.adj
        
        #dropout
        #support = tf.nn.dropout(adj, 1-self.dropout)
        
        support = dot(support, pre_sup, sparse=False)
        
        output = support

        # bias
        if self.bias:
            output += self.weights['bias']

        return self.activation(output)

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
    def __init__(self,placeholders,
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