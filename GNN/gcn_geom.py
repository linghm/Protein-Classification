from __future__ import division
from __future__ import print_function

import time
import os
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import utils
from neural_networks_geom import GCNGraphs




# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'WT', 'which dataset to load')  # PROTEINS
flags.DEFINE_string('dataset_type', 'binary', 'which type of dataset to load')  # PROTEINS

flags.DEFINE_boolean('with_pooling', True, 'whether the mean value for graph labels is computed via pooling(True) or via global nodes(False)')
flags.DEFINE_boolean('local_coord', False, 'whether to use local coordinates system')

flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_integer('batchsize', 100, 'number of graphs for each batch.')
flags.DEFINE_integer('classes', 2, 'number of classes of the dataset.')

flags.DEFINE_integer('hidden2', 8, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden3', 2, 'Number of units in hidden layer2.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('tau', 1.0, 'temperature for softmax.')
flags.DEFINE_float('mu', 1.0, 'weight for the self loop.')

flags.DEFINE_integer('device_id', 0, 'device to use.')
flags.DEFINE_integer('run_id', 0, 'id for run')
flags.DEFINE_integer('seed', 123, 'random seed')
flags.DEFINE_boolean('with_position', True, 'whether use the position features in node features')
flags.DEFINE_integer('patience', 50, 'early stopping patience')

# Set random seed
seed = FLAGS.seed
np.random.seed(seed)
tf.set_random_seed(seed)

tf.disable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device_id)   #指定第一块GPU可用  
config = tf.ConfigProto()  
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存  
config.gpu_options.allow_growth = True      #程序按需申请内存 

print(FLAGS.dataset+FLAGS.dataset_type)
if FLAGS.dataset == 'WT':
    #loading WT data
    dataset_name = "gcnn_data/" + FLAGS.dataset
    features = pkl.load(open(dataset_name+"/"+"ind.HCV_{}_10_ang_aa_energy_7_coord_energyedge_5_hbond.x".format(FLAGS.dataset_type),"rb"))
    edge_features = pkl.load(open(dataset_name+"/"+"ind.HCV_{}_10_ang_aa_energy_7_coord_energyedge_5_hbond.graph".format(FLAGS.dataset_type),"rb"))
    labels = pkl.load(open(dataset_name+"/"+"ind.HCV_{}_10_ang_aa_energy_7_coord_energyedge_5_hbond.y".format(FLAGS.dataset_type),"rb"))
    if FLAGS.dataset_type == 'binary':
        splits = [[0,5300], [5300, 6300], [6300, 7300]]   #total 7342
    elif FLAGS.dataset_type == 'ternary':
        splits = [[0, 6700],[6700,7700],[7700,8700]]    #total 8725
elif FLAGS.dataset == 'A171T' :
    #Loading A171T data
    dataset_name = "gcnn_data/" + FLAGS.dataset
    features = pkl.load(open(dataset_name+"/"+"ind.HCV_A171T_{}_10_ang_aa_energy_7_coord_energyedge_5_hbond.x".format(FLAGS.dataset_type),"rb"))
    edge_features = pkl.load(open(dataset_name+"/"+"ind.HCV_A171T_{}_10_ang_aa_energy_7_coord_energyedge_5_hbond.graph".format(FLAGS.dataset_type),"rb"))
    labels = pkl.load(open(dataset_name+"/"+"ind.HCV_A171T_{}_10_ang_aa_energy_7_coord_energyedge_5_hbond.y".format(FLAGS.dataset_type),"rb"))
    if FLAGS.dataset_type == 'binary':
        splits = [[0,9000], [9000, 11000], [11000, 13000]]
    elif FLAGS.dataset_type == 'ternary':
        splits = [[0,15600],[15600,18600],[18600,21600]]  ##total 21667
elif FLAGS.dataset == 'D183A':
    #Loading D183A data
    dataset_name = "gcnn_data/" + FLAGS.dataset
    features = pkl.load(open(dataset_name+"/"+"ind.HCV_D183A_{}_10_ang_aa_energy_7_coord_energyedge_5_hbond.x".format(FLAGS.dataset_type),"rb"))
    edge_features = pkl.load(open(dataset_name+"/"+"ind.HCV_D183A_{}_10_ang_aa_energy_7_coord_energyedge_5_hbond.graph".format(FLAGS.dataset_type),"rb"))
    labels = pkl.load(open(dataset_name+"/"+"ind.HCV_D183A_{}_10_ang_aa_energy_7_coord_energyedge_5_hbond.y".format(FLAGS.dataset_type),"rb"))
    if FLAGS.dataset_type == 'binary':
        splits = [[0,8500], [8500, 10000], [10000, 11500]]
    elif FLAGS.dataset_type == 'ternary':
        splits = [[0,13600],[13600,15600],[15600,17600]]
elif FLAGS.dataset == 'Triple':
    #Loading Triple data
    dataset_name = "gcnn_data/" + FLAGS.dataset
    features = pkl.load(open(dataset_name+"/"+"ind.HCV_Triple_{}_10_ang_aa_energy_7_coord_energyedge_5_hbond.x".format(FLAGS.dataset_type),"rb"))
    edge_features = pkl.load(open(dataset_name+"/"+"ind.HCV_Triple_{}_10_ang_aa_energy_7_coord_energyedge_5_hbond.graph".format(FLAGS.dataset_type),"rb"))
    labels = pkl.load(open(dataset_name+"/"+"ind.HCV_Triple_{}_10_ang_aa_energy_7_coord_energyedge_5_hbond.y".format(FLAGS.dataset_type),"rb"))
    if FLAGS.dataset_type == 'binary':
        splits = [[0,4800], [4800, 5800], [5800, 6800]]
    elif FLAGS.dataset_type == 'ternary':
        splits = [[0,13000],[13000,15700],[15700,18400]]  ##total 18400
else:
    print('Wrong dataset name')
print(FLAGS.dataset + '{}'.format(splits))
num_graphs = FLAGS.batchsize
num_classes = FLAGS.classes
nodes_per_graph = features.shape[1]
num_nodes = FLAGS.batchsize * nodes_per_graph



#modify here
coords = features[:,:,27:30]
coord = coords[0]

bias_mat = utils.coord_to_bias(coord,local_coord=FLAGS.local_coord)

#node features without position feature

if not FLAGS.with_position:
    print('not using position features')
    features = features[:,:,[i for i in range(27)]+[30]]
else:
    print('using position features')

index = list(range(labels.shape[0]))
np.random.shuffle(index)
train_ind = index[splits[0][0]:splits[0][1]]
val_ind = index[splits[1][0]:splits[1][1]]
test_ind = index[splits[2][0]:splits[2][1]]

#split train,test,val set
features_train = features[train_ind]
edge_features_train = edge_features[train_ind]
labels_train = labels[train_ind]
coords_train = coords[train_ind]

features_val = features[val_ind]
edge_features_val = edge_features[val_ind]
labels_val = labels[val_ind]
coords_val = coords[val_ind]

features_test = features[test_ind]
edge_features_test = edge_features[test_ind]
labels_test = labels[test_ind]
coords_test = coords[test_ind]

idx = np.zeros(num_graphs)
for i in range(num_graphs):
    idx[i] = i * nodes_per_graph

GCN_placeholders = {
    'idx' :tf.placeholder(tf.int32),
    'edge_features': tf.placeholder(tf.float32, shape = (None,
                            edge_features.shape[1],edge_features.shape[2],edge_features.shape[3])),
    'feats': tf.placeholder(tf.float32, shape=(None,features.shape[2])), 
    'coords': tf.placeholder(tf.float32, shape=(None,coords.shape[1], coords.shape[2])), 
    'labels': tf.placeholder(tf.float32, shape=(None, labels.shape[1])),
    'bias_mat': [tf.placeholder(tf.float32, shape=(None,nodes_per_graph,nodes_per_graph)) for _ in range(4)],
    'dropout': tf.placeholder_with_default(0., shape=()),
    }


input_dim = features.shape[2]
print('input_dim:{}'.format(input_dim))
# Create network
network = GCNGraphs(GCN_placeholders, input_dim,  idx, num_graphs, num_nodes, FLAGS.with_pooling)

# Initialize session
sess = tf.Session(config=config)
# Init variables
sess.run(tf.global_variables_initializer())


train_losses = [0. for i in range(0, FLAGS.epochs)]
val_losses = [0. for i in range(0, FLAGS.epochs)]
train_accs = [0. for i in range(0, FLAGS.epochs)]
val_accs = [0. for i in range(0, FLAGS.epochs)]

def evaluate(sess, network, features_test, edge_features_test, labels_test, coords_test, placeholders,bias_mat):
    t_test = time.time()
    total = labels_test.shape[0]
    index = list(range(total))
    acc = 0.
    loss = 0.
    for i in range(0, total, FLAGS.batchsize):
        batch_ind = index[i: (i+FLAGS.batchsize)]
        features_batch = features_test[batch_ind]
        edge_features_batch = edge_features_test[batch_ind]
        labels_batch = labels_test[batch_ind]
        features_batch = np.reshape(features_batch, [features_batch.shape[0]* features_batch.shape[1],
                                                     features_batch.shape[2]])
        features_batch = utils.process_features(features_batch)  
        coords_batch = coords_test[batch_ind]
        
        test_dict = utils.build_dictionary_GCN(features_batch, edge_features_batch, labels_batch,  GCN_placeholders)
        #modify here
        test_dict.update({GCN_placeholders['bias_mat'][i]: bias_mat[i] for i in range(len(bias_mat))})
        test_dict.update({GCN_placeholders['coords']:  coords_batch})
        
        # Test step 
        test_out = sess.run([network.loss, network.accuracy], feed_dict=test_dict)
        acc += FLAGS.batchsize * test_out[1]
        loss += FLAGS.batchsize * test_out[0]
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
checkpt_path = 'pre_trained_geom/{}-{}-epochs{}-batchsize{}-lr{}-hidden2-{}-dropout{}-weight_decay{}-hidden3-{}-mu{}-run{}-seed{}'.format(FLAGS.dataset,FLAGS.dataset_type,FLAGS.epochs,FLAGS.batchsize,
            FLAGS.learning_rate, FLAGS.hidden2,FLAGS.dropout,FLAGS.weight_decay,FLAGS.hidden3,FLAGS.mu,FLAGS.run_id,FLAGS.seed)
if not os.path.exists(checkpt_path):
    os.mkdir(checkpt_path)
checkpt_file = os.path.join(checkpt_path, 'model.ckpt')

for epoch in range(FLAGS.epochs):
    total = labels_train.shape[0]
    index = list(range(total))
    np.random.shuffle(index)
    
    acc = 0.
    loss = 0.
    pos_pred = 0
    for i in range(0, total, FLAGS.batchsize):
        batch_ind = index[i:min(total, i+FLAGS.batchsize)]
        features_batch = features_train[batch_ind]
        edge_features_batch = edge_features_train[batch_ind]
        
        labels_batch = labels_train[batch_ind]
        features_batch = np.reshape(features_batch, [features_batch.shape[0]* features_batch.shape[1],
                                                     features_batch.shape[2]])
        features_batch = utils.process_features(features_batch)
        coords_batch = coords_train[batch_ind]
        
        train_dict = utils.build_dictionary_GCN(features_batch, edge_features_batch, labels_batch,  GCN_placeholders)
        #modify here
        train_dict.update({GCN_placeholders['bias_mat'][i]: bias_mat[i] for i in range(len(bias_mat))})
        train_dict.update({GCN_placeholders['dropout']: FLAGS.dropout})
        train_dict.update({GCN_placeholders['coords']:  coords_batch})
        # Training step 
        train_out = sess.run([network.opt_op, network.loss, network.accuracy, network.pos_pred], feed_dict=train_dict)
        
        acc += FLAGS.batchsize * train_out[2]
        loss += FLAGS.batchsize * train_out[1]
        pos_pred += train_out[3]
        
    acc /= total
    loss /=total
    
    train_losses[epoch] = loss
    train_accs[epoch] = acc
    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss), "train_acc=", "{:.5f}".format(acc), "pos_pred:{}".format(pos_pred))
    
    # validation
    val_loss, val_acc, val_duration = evaluate(sess, network, features_val, edge_features_val, labels_val,coords_val, GCN_placeholders,bias_mat)
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
test_loss, test_acc, test_duration = evaluate(sess, network, features_test, edge_features_test, labels_test,coords_test, GCN_placeholders,bias_mat)
print("Epoch:", '%04d' % (epoch_early_model+1), "test_loss=", "{:.5f}".format(test_loss), "test_acc=", "{:.5f}".format(test_acc))

print(time.time()-t)
filename = '_result_geom_earlystop.txt'
with open(filename,'a') as f: 
    f.write("##{}-{} geom: with_position:{},epochs:{},batchsize:{},lr:{},hidden2:{},dropout:{},weight_decay:{},hidden3:{},mu:{},local_coord:{},seed:{}\n".format(FLAGS.dataset,FLAGS.dataset_type,FLAGS.with_position,FLAGS.epochs,FLAGS.batchsize,
            FLAGS.learning_rate, FLAGS.hidden2,FLAGS.dropout,FLAGS.weight_decay,FLAGS.hidden3, FLAGS.mu, FLAGS.local_coord,FLAGS.seed))
    f.write('best_epoch:{}, best_val_acc:{}, best_val_loss:{}\n'.format(
            epoch_early_model+1,vacc_early_model,vlss_early_model))
    f.write("test_loss:{},test_acc:{}\n".format(test_loss,test_acc))



