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

    def _call(self, edge_features,inputs):
        raise NotImplementedError

    def __call__(self, edge_features, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(edge_features,inputs)
            return outputs
            
class ConvolutionalLayer(Layer):
    def __init__(self, input_dim, output_dim, placeholders, dropout,
                 sparse_inputs, activation,  bias=False,  **kwargs):
        super(ConvolutionalLayer, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        
        self.activation = activation
        
        self.sparse_inputs = sparse_inputs
        self.bias = bias
       
        self.num_edge_feature = 8
        self.weights['coef'] = tf.get_variable('coef', shape=[self.num_edge_feature,1],
                    initializer=tf.random_normal_initializer())
        '''
        
        self.weights['fc1'] = tf.get_variable('fc1', shape=[9, 9],
                    initializer=tf.random_normal_initializer())
        self.weights['fc2'] = tf.get_variable('fc2', shape=[9, 1],
                    initializer=tf.random_normal_initializer())
        '''
        with tf.variable_scope(self.name + '_weights'):
            '''
            self.weights['coef_node1'] = tf.get_variable('coef_node1', shape=[output_dim,1],
                    initializer=tf.random_normal_initializer())
            self.weights['coef_node2'] = tf.get_variable('coef_node2', shape=[output_dim,1],
                    initializer=tf.random_normal_initializer())
            '''
            self.weights['weights'] = glorot([input_dim, output_dim],
                                                        name='weights')
            if self.bias:
                self.weights['bias'] = zeros([output_dim], name='bias')

    def _call(self, edge_features, inputs):
        x = inputs

        # dropout
        
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)
        
        # convolve
        
        pre_sup = dot(x, self.weights['weights'],
                      sparse=self.sparse_inputs)
        
        
        # construct support(normalized laplcian by edge_features)
        num_nodes = edge_features.get_shape().as_list()[1]
        
        
        edge_features = tf.reshape(edge_features,[FLAGS.batchsize * num_nodes *num_nodes, self.num_edge_feature])
        edge_features = tf.matmul(edge_features, self.weights['coef'])
        edge_features = tf.reshape(edge_features, [FLAGS.batchsize, num_nodes, num_nodes])
        edge_features = tf.nn.softmax(edge_features/FLAGS.tau)
        '''
        edge_features = tf.reshape(self.edge_features,[FLAGS.batchsize * 40 *40, 9])
        edge_features = tf.nn.relu(tf.matmul(edge_features, self.weights['fc1']) + edge_features)
        edge_features = tf.nn.relu(tf.matmul(edge_features, self.weights['fc2']))
        edge_features = tf.reshape(edge_features, [FLAGS.batchsize, 40, 40])
        edge_features = tf.nn.leaky_relu(edge_features)
        edge_features = tf.nn.softmax(edge_features/FLAGS.tau)
        '''
        '''
        edge_features = tf.reshape(self.edge_features,[FLAGS.batchsize * 40 *40, 129])
        edge_features = tf.nn.relu(tf.matmul(edge_features, self.weights['fc1']))
        edge_features = tf.nn.relu(tf.matmul(edge_features, self.weights['fc2']))
        edge_features = tf.reshape(edge_features, [FLAGS.batchsize, 40, 40])
        '''
        '''
        edge_features = tf.reshape(self.edge_features,[FLAGS.batchsize * 40 *40, 9])
        edge_features = tf.matmul(edge_features, self.weights['coef'])
        edge_features = tf.reshape(edge_features, [FLAGS.batchsize, 40, 40])
        f1 = tf.matmul(pre_sup,self.weights['coef_node1'])
        f2 = tf.matmul(pre_sup,self.weights['coef_node2'])
        f1 = tf.reshape(f1,[FLAGS.batchsize,40,1])
        f2 = tf.reshape(f2,[FLAGS.batchsize,1,40])
        attention = f1 + f2
        edge_features = tf.nn.leaky_relu(attention + edge_features)
        edge_features = tf.nn.softmax(edge_features/FLAGS.tau)
        '''
        adj = block_diagonal(edge_features)
        support = preprocess_adj(adj, True, False)
        #support= adj
        
        #dropout
        #support = tf.nn.dropout(adj, 1-self.dropout)
        
        support = dot(support, pre_sup, sparse=False)
        
        output = support

        # bias
        if self.bias:
            output += self.weights['bias']

        return self.activation(output)

class PoolingLayer(Layer):
    def __init__(self, num_graphs, num_nodes, idx,placeholders,
                 activation,  bias=False,  **kwargs):
        super(PoolingLayer, self).__init__(**kwargs)

        self.num_nodes = num_nodes
        self.num_graphs = num_graphs
        self.activation = activation
        self.idx = idx

    def _call(self, inputs):
        
        pooling_matrix = np.array([[0. for i in range(self.num_nodes)] for k in range(self.num_graphs)])
        idx_aug = np.append(self.idx, self.num_nodes)
        idx_aug = idx_aug.astype(int)


        for i in range(self.num_graphs):
            pooling_matrix[i, range(idx_aug[i], idx_aug[i+1])] = (1/(idx_aug[i+1]-idx_aug[i]))

        output = dot(tf.cast(pooling_matrix, tf.float32), inputs, sparse = False)
    

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