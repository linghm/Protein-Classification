from __future__ import division
from __future__ import print_function
import scipy.sparse as sp
import time
import os
import tensorflow.compat.v1 as tf
import numpy as np
import pickle as pkl
import utils
import random
#from neural_networks_geom import GCNGraphs
from neural_networks_geom_newdata import GCNGraphs





# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'bbbp', 'which dataset to load')  # PROTEINS

flags.DEFINE_boolean('with_pooling', True, 'whether the mean value for graph labels is computed via pooling(True) or via global nodes(False)')
flags.DEFINE_boolean('local_coord', False, 'whether to use local coordinates system')

flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('batchsize', 64, 'number of graphs for each batch.')
flags.DEFINE_integer('classes', 2, 'number of classes of the dataset.')

flags.DEFINE_integer('hidden2', 8, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden3', 2, 'Number of units in hidden layer2.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('tau', 1.0, 'temperature for softmax.')
flags.DEFINE_float('mu', 1.0, 'weight for the self loop.')

flags.DEFINE_integer('device_id', 0, 'device to use.')
flags.DEFINE_integer('run_id', 0, 'id for run')
flags.DEFINE_boolean('with_position', True, 'whether use the position features in node features')
flags.DEFINE_integer('patience', 50, 'early stopping patience')


tf.disable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device_id)   #指定第一块GPU可用  
config = tf.ConfigProto()  
#config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存  
config.gpu_options.allow_growth = True      #程序按需申请内存 

with open("./data/" + FLAGS.dataset + "_train_processed.txt", "rb") as f_in_train:
        train_data_processed = pkl.load(f_in_train)

with open("./data/" + FLAGS.dataset + "_val_processed.txt", "rb") as f_in_val:
        val_data_processed = pkl.load(f_in_val)

with open("./data/" + FLAGS.dataset + "_test_processed.txt", "rb") as f_in_test:
        test_data_processed = pkl.load(f_in_test)




input_dim = train_data_processed[0][0].shape[1]
print('input_dim:{}'.format(input_dim))
GCN_placeholders = {
    'adj': tf.placeholder(tf.float32, shape = (None,None)),
    'dist_adj': tf.placeholder(tf.float32, shape = (None,None)),
    'pooling_matrix': tf.placeholder(tf.float32, shape=(None,None)),
    'feats': tf.placeholder(tf.float32, shape=(None,input_dim)), 
    'coords': tf.placeholder(tf.float32, shape=(None,3)), 
    'labels': tf.placeholder(tf.float32, shape=(None,FLAGS.classes)),
    'bias_mat': [tf.placeholder(tf.float32, shape=(None,None)) for _ in range(4)],
    'dropout': tf.placeholder_with_default(0., shape=()),
    }

# Create network
network = GCNGraphs(GCN_placeholders, input_dim,  FLAGS.with_pooling)

# Initialize session
sess = tf.Session(config=config)
# Init variables
sess.run(tf.global_variables_initializer())


train_losses = [0. for i in range(0, FLAGS.epochs)]
val_losses = [0. for i in range(0, FLAGS.epochs)]
train_accs = [0. for i in range(0, FLAGS.epochs)]
val_accs = [0. for i in range(0, FLAGS.epochs)]
def build_dictionary_GCN(feats, adj, dist_adj, labels, bias_mat, coords, pooling_matrix, placeholders):
    #prepara il dizionario che sarà poi passato alla sessione di TF
    dictionary = dict()
    dictionary.update({placeholders['feats']: feats})
    dictionary.update({placeholders['adj']: adj})
    dictionary.update({placeholders['dist_adj']: dist_adj})
    dictionary.update({placeholders['labels']: labels})
    dictionary.update({placeholders['bias_mat'][i]: bias_mat[i] for i in range(len(bias_mat))})
    dictionary.update({placeholders['coords']:  coords_batch})
    dictionary.update({placeholders['pooling_matrix']: pooling_matrix})

    return dictionary

def batch_distance_matrix(batch_coords, nodes_size_list):
    nodes_size_cumsum_list = np.cumsum(nodes_size_list)
    dist_adj = np.zeros([nodes_size_cumsum_list[-1],nodes_size_cumsum_list[-1]])
    for index in range(len(nodes_size_list)-1):
        coords = batch_coords[nodes_size_cumsum_list[index]:
                              nodes_size_cumsum_list[index+1], :]
        r = np.sum(coords * coords, axis=1, keepdims=True)
        m = np.matmul(coords, np.transpose(coords))
        D = r - 2 * m + np.transpose(r)
        dist_adj[nodes_size_cumsum_list[index]:nodes_size_cumsum_list[index+1],
                 nodes_size_cumsum_list[index]:nodes_size_cumsum_list[index+1]] = D
    return dist_adj

def evaluate(sess, network, test_data_processed, placeholders):
    t_test = time.time()
    total = len(test_data_processed)
    acc = 0.
    loss = 0.
    for i in range(0, total, FLAGS.batchsize):
        graph_batch = test_data_processed[i: min(total,i+FLAGS.batchsize)]
        features_batch = [graph_batch[index][0] for index in range(len(graph_batch))]
        adjs_batch = [graph_batch[index][1].toarray() for index in range(len(graph_batch))]        
        labels_batch = [graph_batch[index][2].reshape(1,-1) for index in range(len(graph_batch))]
        coords_batch = [graph_batch[index][3] for index in range(len(graph_batch))]
        
        features_batch = np.concatenate(features_batch,axis=0)        
        features_batch = utils.process_features(features_batch)
        labels_batch = np.concatenate(labels_batch,axis=0)
        coords_batch = np.concatenate(coords_batch, axis=0)
        #construct a diagonal adj matrix for a batch of graphs
        nodes_size_list = [0] + [adjs_batch[index].shape[0] for index in range(len(graph_batch))]
        nodes_size_cumsum_list = np.cumsum(nodes_size_list)
        adj = np.zeros([nodes_size_cumsum_list[-1], nodes_size_cumsum_list[-1]])
        for index in range(len(adjs_batch)):
            adj[nodes_size_cumsum_list[index]:nodes_size_cumsum_list[index+1],
                nodes_size_cumsum_list[index]:nodes_size_cumsum_list[index+1]] = adjs_batch[index]
        dist_adj = batch_distance_matrix(coords_batch,nodes_size_list)
        
        # we don't normalize the adjacent matrix here 
        pooling_matrix = np.zeros([len(graph_batch),nodes_size_cumsum_list[-1]])
        for i in range(len(graph_batch)):
            pooling_matrix[i, range(nodes_size_cumsum_list[i], nodes_size_cumsum_list[i+1])] = (1./(nodes_size_cumsum_list[i+1] - nodes_size_cumsum_list[i]))
        # construct the bias matrix
        bias_mat = utils.batch_coord_to_bias(coords_batch , nodes_size_list, local_coord=FLAGS.local_coord )

        test_dict = build_dictionary_GCN(features_batch, adj, dist_adj, labels_batch, bias_mat, coords_batch, pooling_matrix, placeholders)
        
        # Test step 
        test_out = sess.run([network.loss, network.accuracy], feed_dict=test_dict)
        acc += len(graph_batch) * test_out[1]
        loss += len(graph_batch) * test_out[0]
    acc /= total
    loss /=total
    return loss, acc, (time.time() - t_test)

# Train network using labels_train, features_train, edge_features_train
t = time.time()
# early stopping
saver = tf.train.Saver()
vlss_mn = np.inf
vacc_mx = 0.0
curr_step = 0
checkpt_path = 'pre_trained_geom/{}-epochs{}-batchsize{}-lr{}-hidden2-{}-dropout{}-weight_decay{}-hidden3-{}-mu{}-run{}'.format(FLAGS.dataset,FLAGS.epochs,FLAGS.batchsize,
            FLAGS.learning_rate, FLAGS.hidden2,FLAGS.dropout,FLAGS.weight_decay,FLAGS.hidden3,FLAGS.mu,FLAGS.run_id)
if not os.path.exists(checkpt_path):
    os.mkdir(checkpt_path)
checkpt_file = os.path.join(checkpt_path, 'model.ckpt')

total = len(train_data_processed)
for epoch in range(FLAGS.epochs):    
    random.shuffle(train_data_processed)
    acc = 0.
    loss = 0.
    pos_pred = 0
    for i in range(0, total, FLAGS.batchsize):
        graph_batch = train_data_processed[i: min(total,i+FLAGS.batchsize)]
        features_batch = [graph_batch[index][0] for index in range(len(graph_batch))]
        adjs_batch = [graph_batch[index][1].toarray() for index in range(len(graph_batch))]        
        labels_batch = [graph_batch[index][2].reshape(1,-1) for index in range(len(graph_batch))]
        coords_batch = [graph_batch[index][3] for index in range(len(graph_batch))]
        
        features_batch = np.concatenate(features_batch,axis=0)        
        features_batch = utils.process_features(features_batch)
        labels_batch = np.concatenate(labels_batch,axis=0)
        coords_batch = np.concatenate(coords_batch, axis=0)
        #construct a diagonal adj matrix for a batch of graphs
        nodes_size_list = [0] + [adjs_batch[index].shape[0] for index in range(len(graph_batch))]
        nodes_size_cumsum_list = np.cumsum(nodes_size_list)
        adj = np.zeros([nodes_size_cumsum_list[-1], nodes_size_cumsum_list[-1]])
        for index in range(len(adjs_batch)):
            adj[nodes_size_cumsum_list[index]:nodes_size_cumsum_list[index+1],
                nodes_size_cumsum_list[index]:nodes_size_cumsum_list[index+1]] = adjs_batch[index]
        dist_adj = batch_distance_matrix(coords_batch,nodes_size_list)
        # we don't normalize the adjacent matrix here 
        
        pooling_matrix = np.zeros([len(graph_batch),nodes_size_cumsum_list[-1]])
        for i in range(len(graph_batch)):
            pooling_matrix[i, range(nodes_size_cumsum_list[i], nodes_size_cumsum_list[i+1])] = (1./(nodes_size_cumsum_list[i+1] - nodes_size_cumsum_list[i]))
        # construct the bias matrix
        bias_mat = utils.batch_coord_to_bias(coords_batch , nodes_size_list, local_coord=FLAGS.local_coord )
        
        train_dict = build_dictionary_GCN(features_batch, adj, dist_adj, labels_batch, bias_mat, coords_batch, pooling_matrix, GCN_placeholders)
        #modify here
        train_dict.update({GCN_placeholders['dropout']: FLAGS.dropout})
        
        # Training step 
        train_out = sess.run([network.opt_op, network.loss, network.accuracy, network.pos_pred], feed_dict=train_dict)
        
        acc += len(graph_batch) * train_out[2]
        loss += len(graph_batch) * train_out[1]
        pos_pred += train_out[3]
        
    acc /= total
    loss /=total
    
    train_losses[epoch] = loss
    train_accs[epoch] = acc
    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss), "train_acc=", "{:.5f}".format(acc), "pos_pred:{}".format(pos_pred))
   
    # validation
    val_loss, val_acc, val_duration = evaluate(sess, network, val_data_processed, GCN_placeholders)
    print("Epoch:", '%04d' % (epoch + 1), "val_loss=", "{:.5f}".format(val_loss), "val_acc=", "{:.5f}".format(val_acc))
    val_losses[epoch] = val_loss
    val_accs[epoch] = val_acc
    
    # early stopping
    if val_acc >= vacc_mx or val_loss<= vlss_mn:
        if val_acc>= vacc_mx and val_loss<= vlss_mn:
            vacc_early_model = val_acc
            vlss_early_model = val_loss
            epoch_early_model = epoch
            saver.save(sess, checkpt_file)
        vacc_mx = np.max((val_acc, vacc_mx))
        vlss_mn = np.min((val_loss, vlss_mn))
        curr_step = 0
    else:
        curr_step += 1
        if curr_step == FLAGS.patience:
            print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
            print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
            print(time.time()-t)
            break    
    
print("Optimization Finished!")

#testing
print('Testing')
saver.restore(sess, checkpt_file)
test_loss, test_acc, test_duration = evaluate(sess, network, test_data_processed , GCN_placeholders)
print("Epoch:", '%04d' % (epoch_early_model+1), "test_loss=", "{:.5f}".format(test_loss), "test_acc=", "{:.5f}".format(test_acc))

print(time.time()-t)
filename = 'newdata_result_geom_earlystop.txt'
with open(filename,'a') as f: 
    f.write("##{} geom: with_position:{},epochs:{},batchsize:{},lr:{},hidden2:{},dropout:{},weight_decay:{},hidden3:{},mu:{},local_coord:{}\n".format(FLAGS.dataset,FLAGS.with_position,FLAGS.epochs,FLAGS.batchsize,
            FLAGS.learning_rate, FLAGS.hidden2,FLAGS.dropout,FLAGS.weight_decay,FLAGS.hidden3, FLAGS.mu, FLAGS.local_coord))
    f.write('earlystop_epoch:{}, earlystop_val_acc:{}, earlystop_val_loss:{}\n'.format(
            epoch_early_model+1,vacc_early_model,vlss_early_model))
    f.write("test_loss:{},test_acc:{}\n".format(test_loss,test_acc))



