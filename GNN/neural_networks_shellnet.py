from layers_shellnet import ConvolutionalLayer,  ShellConvLayer
import utils
import tensorflow.compat.v1 as tf
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS

_GRAPH_UIDS = {}
def get_graph_uid(graph_name=''):
    if graph_name not in _GRAPH_UIDS:
        _GRAPH_UIDS[graph_name] = 1
        return 1
    else:
        _GRAPH_UIDS[graph_name] += 1
        return _GRAPH_UIDS[graph_name]
    
def dense(input, output, is_training, name, bn_decay=None, with_bn=False, activation=tf.nn.relu):
    if with_bn:
        input = tf.layers.batch_normalization(input, momentum=0.98, training=is_training, name=name+'bn')
    
    dense = tf.layers.dense(input, output, activation=activation, name=name)
    
    return dense

class BaseNet(object):
    def __init__(self, **kwargs):
        graph = self.__class__.__name__.lower()
        name = graph + '_' + str(get_graph_uid(graph))
        self.name = name
        self.weights = {}
        self.placeholders = {}
        self.is_training = None
        self.layers = []
        self.activations = []
        self.inputs = None
        self.outputs = None
        self.loss = 0
        self.accuracy = 0
        self.pos_pred = 0
        #modify here
        self.points = None
        self.edge_features = None
        
        self.optimizer = None
        self.opt_op = None
    def _build(self):
        raise NotImplementedError

    def build(self):
        
        
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self._build()
        self.activations.append(self.inputs)
        
        hidden = self.layers[0](self.edge_features, self.activations[-1])
        self.activations.append(hidden)
        
        # nodes coords for shellnet
        points = self.points
        # edge coords for shellnet
        edge_points_1 = tf.reshape(points,(FLAGS.batchsize,points.get_shape().as_list()[1],1,3))
        edge_points_1 = tf.tile(edge_points_1,(1,1, points.get_shape().as_list()[1],1))       
        edge_points_2 = tf.reshape(points,(FLAGS.batchsize,1,points.get_shape().as_list()[1],3))
        edge_points_2 = tf.tile(edge_points_2,(1, points.get_shape().as_list()[1],1,1))        
        edge_points = tf.concat([edge_points_1,edge_points_2],axis=3)
        edge_points = tf.reshape(edge_points, (FLAGS.batchsize,points.get_shape().as_list()[1]**2,6))
        
        P = 16
        #queries = tf.slice(points, (0, 0, 0), (-1, P, -1), name='sconv1_queries')
        index = list(range(34))
        np.random.shuffle(index)
        #sample_index = [ 0,  1,  2,  3,  8, 12, 13, 14, 17, 21, 23, 26, 28, 29, 31, 32]
        sample_index = index[:P]
        
        queries_indices = tf.tile(tf.reshape(sample_index,(1,P)),(FLAGS.batchsize,1))
        batch_indices = tf.tile(tf.reshape(tf.range(FLAGS.batchsize), (-1, 1, 1)), (1, P, 1)) #(N,P,1)
        indices = tf.concat([batch_indices, tf.expand_dims(queries_indices, axis=2)], axis=2)
        # node queries for shellnet
        queries = tf.gather_nd(points, indices, name='sconv1_queries')
        #edge queries for shellnet
        edge_queries_1 = tf.reshape(queries,(FLAGS.batchsize,queries.get_shape().as_list()[1],1,-1))
        edge_queries_1 = tf.tile(edge_queries_1,(1,1, queries.get_shape().as_list()[1],1))       
        edge_queries_2 = tf.reshape(queries,(FLAGS.batchsize,1,queries.get_shape().as_list()[1],-1))
        edge_queries_2 = tf.tile(edge_queries_2,(1, queries.get_shape().as_list()[1],1,1))        
        edge_queries = tf.concat([edge_queries_1,edge_queries_2],axis=3)
        edge_queries = tf.reshape(edge_queries, (FLAGS.batchsize,queries.get_shape().as_list()[1]**2,-1))
        
        
        
        # previous edge features
        edge_features_prev = tf.reshape(self.edge_features,(FLAGS.batchsize,self.edge_features.get_shape().as_list()[1]**2,8))
        #previous node features 
        features_prev = tf.reshape(self.activations[-1],[FLAGS.batchsize, -1, 
                                   self.activations[-1].get_shape().as_list()[1]])
        hidden =  self.layers[2](points, features_prev, queries)     # batchsize*P*C
        hidden = tf.reshape(hidden,(FLAGS.batchsize*P, -1))
        self.activations.append(hidden)
        
        edge_features = self.layers[3](edge_points, edge_features_prev, edge_queries)
        edge_features = tf.reshape(edge_features,(FLAGS.batchsize,queries.get_shape().as_list()[1],queries.get_shape().as_list()[1],-1))
        
        hidden = self.layers[1](edge_features, self.activations[-1])
        self.activations.append(hidden)
        
        
        
        points = queries
        
        #queries = tf.slice(points, (0, 0, 0), (-1, P, -1), name='sconv2_queries')
        index = list(range(P))
        np.random.shuffle(index)
        P = 1
        sample_index = index[:P]
        #sample_index = [8]
        queries_indices = tf.tile(tf.reshape(sample_index,(1,P)),(FLAGS.batchsize,1))
        batch_indices = tf.tile(tf.reshape(tf.range(FLAGS.batchsize), (-1, 1, 1)), (1, P, 1)) #(N,P,1)
        indices = tf.concat([batch_indices, tf.expand_dims(queries_indices, axis=2)], axis=2)
        queries = tf.gather_nd(points, indices, name='sconv2_queries')
        
        features_prev = self.activations[-1]
        features_prev = tf.reshape(features_prev, (FLAGS.batchsize, 16,-1))
        hidden =  self.layers[-1](points, features_prev, queries)
        self.activations.append(hidden)
        
        hidden = dense(self.activations[-1], output = FLAGS.classes, is_training=self.is_training, name='logits', activation=lambda x: x)
        hidden = tf.reshape(hidden, [FLAGS.batchsize * P,hidden.get_shape().as_list()[2]])
        self.activations.append(hidden)
        
        self.outputs = self.activations[-1]
        #这里记录了所有的tf variable(global)
        self.weights = {var.name: var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)}

        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError






class GCNGraphs(BaseNet):
    def __init__(self, placeholders, input_dim,  idx, num_graphs, num_nodes, with_pooling, **kwargs):
        super(GCNGraphs, self).__init__(**kwargs)
        
        #modify here
        self.points = placeholders['points']
        self.edge_features = placeholders['edge_features']
        
        self.pooling = with_pooling
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.idx = idx
        self.inputs = placeholders['feats']
        self.input_dim = input_dim
        #modify here
        
        self.placeholders = placeholders
        self.is_training = placeholders['is_training']
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].weights.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        #cross entropy loss dopo aver applicato un softmax layer
        self.loss += utils.cross_entropy(self.outputs, self.placeholders['labels'])
        #self.loss += utils.binary_cross_entropy(self.outputs, self.placeholders['labels'])

    def _accuracy(self):
        self.accuracy , self.pos_pred= utils.accuracy_cal(self.outputs, self.placeholders['labels'])
        #self.accuracy= utils.binary_accuracy_cal(self.outputs, self.placeholders['labels'])
    
    def _build(self): 
        self.layers.append(ConvolutionalLayer(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            activation=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=False
                                            ))


        
        
        self.layers.append(ConvolutionalLayer(input_dim=FLAGS.hidden2,
                                            output_dim=FLAGS.hidden3,
                                            placeholders=self.placeholders,
                                            activation=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=False
                                            ))
        '''
        if self.pooling:
            self.layers.append(PoolingLayer(    num_graphs = self.num_graphs,
                                                num_nodes = self.num_nodes,
                                                idx=self.idx,
                                                placeholders=self.placeholders,
                                                activation=lambda x: x
                                                
                                                ))
        
        '''
        
        self.layers.append(ShellConvLayer(is_training=self.is_training, tag = 'shellconv1', K =8, D=2, P=16, C=FLAGS.hidden2, with_local=FLAGS.with_local)) # for node pooling
        self.layers.append(ShellConvLayer(is_training=self.is_training, tag = 'shellconv2', K =64, D=4, P=256, C=8, with_local=FLAGS.with_local))   # for edge pooling
        self.layers.append(ShellConvLayer(is_training=self.is_training, tag = 'shellconv3', K =16, D=4, P=1, C=64, with_local=FLAGS.with_local))
    
    def predict(self):
        return tf.nn.softmax(self.outputs)