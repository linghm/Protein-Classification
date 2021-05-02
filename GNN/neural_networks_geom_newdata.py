from geom_layers_newdata import ConvolutionalLayer, PoolingLayer, DenseLayer
import utils
import tensorflow.compat.v1 as tf
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

class BaseNet(object):
    def __init__(self, **kwargs):
        graph = self.__class__.__name__.lower()
        name = graph + '_' + str(get_graph_uid(graph))
        self.name = name
        self.weights = {}
        self.placeholders = {}
        self.layers = []
        self.activations = []
        self.inputs = None
        self.outputs = None
        self.loss = 0
        self.accuracy = 0
        self.pos_pred = 0
        self.optimizer = None
        self.opt_op = None
    def _build(self):
        raise NotImplementedError

    def build(self):
       
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self._build()
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
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
    def __init__(self, placeholders, input_dim,  with_pooling, **kwargs):
        super(GCNGraphs, self).__init__(**kwargs)

        self.pooling = with_pooling
        self.inputs = placeholders['feats']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].weights.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        #cross entropy loss dopo aver applicato un softmax layer
        self.loss += utils.cross_entropy(self.outputs, self.placeholders['labels'])
        
    def _accuracy(self):
        self.accuracy , self.pos_pred= utils.accuracy_cal(self.outputs, self.placeholders['labels'])
        
    def _build(self): 
        self.layers.append(ConvolutionalLayer(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            activation=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=False
                                            ))


        
        
        self.layers.append(ConvolutionalLayer(input_dim=FLAGS.hidden2*8,
                                            #output_dim=self.output_dim,
                                            output_dim = FLAGS.hidden3,
                                            placeholders=self.placeholders,
                                            #activation=lambda x: x,
    
                                            activation = tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=False
                                            ))
      
        self.layers.append(DenseLayer(output_dim = FLAGS.classes,                                      
                                      activation=lambda x: x ,
                                        placeholders=self.placeholders,
                                        dropout=True                                               
                                          ))
        
        if self.pooling:
            self.layers.append(PoolingLayer(   
                                                placeholders=self.placeholders,
                                                activation=lambda x: x
                                                
                                                ))
        




    def predict(self):
        return tf.nn.softmax(self.outputs)