import numpy as np
import time
import scipy.sparse as sp
import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

#generate bias matrix by coords 
def coord_to_bias(coord,local_coord=False):
    #保证平移不变性和旋转不变性
    nodes_per_graph = coord.shape[0]
    '''
    coord_mat = np.zeros([3,3])
    coord_mat[0,:] = coord[1,:]-coord[0,:]
    coord_mat[1,:] = coord[2,:]-coord[0,:]
    coord_mat[2,:] = coord[3,:]-coord[0,:]
    coord = coord - coord[0,:]
    coord = np.matmul(coord,np.linalg.inv(coord_mat))
    '''
    x1 = coord[:,0]
    y1 = coord[:,1]
    z1 = coord[:,2]
    direc_index = []
    
    for i in range(nodes_per_graph):
        temp = [[],[],[],[]]
        if local_coord:
            coord_mat = np.zeros([3,3])
            if i>0 and i< nodes_per_graph-1:                
                coord_mat[0,:] = coord[i-1,:]-coord[i,:]
                coord_mat[1,:] = coord[i+1,:]-coord[i,:]
            elif i==0:
                coord_mat[0,:] = coord[i+1,:]-coord[i,:]
                coord_mat[1,:] = coord[i+2,:]-coord[i,:]
            elif i==nodes_per_graph-1:
                coord_mat[0,:] = coord[i-2,:]-coord[i,:]
                coord_mat[1,:] = coord[i-1,:]-coord[i,:]
            coord_mat[2,:] = np.cross(coord_mat[0,:],coord_mat[1,:])
            
            coord_temp = coord - coord[i,:]         
            coord_temp = np.matmul(coord_temp,np.linalg.inv(coord_mat))
            x1 = coord_temp[:,0]
            y1 = coord_temp[:,1]
            z1 = coord_temp[:,2]
                    
        for j in range(nodes_per_graph):
            if x1[j] > x1[i] and y1[j] > y1[i]:
                temp[0].append(j)
            elif x1[j] > x1[i] and y1[j] < y1[i]:
                temp[1].append(j)
            elif x1[j] < x1[i] and y1[j] < y1[i]:
                temp[2].append(j)
            elif x1[j] < x1[i] and y1[j] > y1[i] :
                temp[3].append(j)
        direc_index.append(temp)
    
    adjs = [np.zeros([FLAGS.batchsize,nodes_per_graph,nodes_per_graph]) for i in range(4)]
    for i in range(FLAGS.batchsize):
        for j in range(nodes_per_graph):
            adjs[0][i][j][direc_index[j][0]] = 1.
            adjs[1][i][j][direc_index[j][1]] = 1.
            adjs[2][i][j][direc_index[j][2]] = 1.
            adjs[3][i][j][direc_index[j][3]] = 1.
    bias_mat = [adj_to_bias(adjs[i],sizes=[nodes_per_graph]*FLAGS.batchsize,nhood=1) for i in range(len(adjs))]
    return bias_mat
def batch_coord_to_bias(coords_batch ,nodes_size_list,local_coord=False):
    nodes_size_cumsum_list = np.cumsum(nodes_size_list)
    batch_adjs = []
    for index in range(len(nodes_size_list)-1): 
        coord = coords_batch[nodes_size_cumsum_list[index]:nodes_size_cumsum_list[index+1]]
        #保证平移不变性和旋转不变性
        num_nodes = coord.shape[0]
        x1 = coord[:,0]
        y1 = coord[:,1]
        z1 = coord[:,2]
        direc_index = []
    
        for i in range(num_nodes):
            temp = [[],[],[],[]]
            if local_coord:
                coord_mat = np.zeros([3,3])
                if i>0 and i< num_nodes - 1:                
                    coord_mat[0,:] = coord[i-1,:]-coord[i,:]
                    coord_mat[1,:] = coord[i+1,:]-coord[i,:]
                elif i==0:
                    coord_mat[0,:] = coord[i+1,:]-coord[i,:]
                    coord_mat[1,:] = coord[i+2,:]-coord[i,:]
                elif i==num_nodes-1:
                    coord_mat[0,:] = coord[i-2,:]-coord[i,:]
                    coord_mat[1,:] = coord[i-1,:]-coord[i,:]
                coord_mat[2,:] = np.cross(coord_mat[0,:],coord_mat[1,:])
                
                coord_temp = coord - coord[i,:]         
                coord_temp = np.matmul(coord_temp,np.linalg.inv(coord_mat))
                x1 = coord_temp[:,0]
                y1 = coord_temp[:,1]
                z1 = coord_temp[:,2]
                        
            for j in range(num_nodes):
                if x1[j] > x1[i] and y1[j] > y1[i]:
                    temp[0].append(j)
                elif x1[j] > x1[i] and y1[j] < y1[i]:
                    temp[1].append(j)
                elif x1[j] < x1[i] and y1[j] < y1[i]:
                    temp[2].append(j)
                elif x1[j] < x1[i] and y1[j] > y1[i] :
                    temp[3].append(j)
            direc_index.append(temp)
        
        temp_adjs = [np.zeros([num_nodes,num_nodes]) for i in range(4)]
        for i in range(num_nodes):
            temp_adjs[0][i][direc_index[i][0]] = 1.
            temp_adjs[1][i][direc_index[i][1]] = 1.
            temp_adjs[2][i][direc_index[i][2]] = 1.
            temp_adjs[3][i][direc_index[i][3]] = 1.
        batch_adjs.append(temp_adjs)
    adjs = [np.zeros([nodes_size_cumsum_list[-1],nodes_size_cumsum_list[-1]]) for i in range(4)]
    for index in range(len(nodes_size_list)-1):
        for i in range(4):
            adjs[i][nodes_size_cumsum_list[index]:nodes_size_cumsum_list[index+1],
                    nodes_size_cumsum_list[index]:nodes_size_cumsum_list[index+1]] = batch_adjs[index][i]
    bias_mat = [batch_adj_to_bias(adjs[i], nodes_size_list) for i in range(len(adjs))]
    return bias_mat
def batch_adj_to_bias(adj, nodes_size_list):
    mt = adj + np.eye(adj.shape[1])
    nodes_size_cumsum_list = np.cumsum(nodes_size_list)
    for index in range(len(nodes_size_list)-1):        
        for i in range(nodes_size_list[index+1]):
            for j in range(nodes_size_list[index+1]):
                if mt[nodes_size_cumsum_list[index]+i, nodes_size_cumsum_list[index]+j ] > 0.0:
                    mt[nodes_size_cumsum_list[index]+i, nodes_size_cumsum_list[index]+j ] = 1.0
    #这一步很关键，乘上-1e9这个很大的负数再作softmax会趋于零。
    return -1e9 * (1.0 - mt)

def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    #这一步很关键，乘上-1e9这个很大的负数再作softmax会趋于零。
    return -1e9 * (1.0 - mt)

#trasforma matrici in tuple
def to_tuple(mat):
    if not sp.isspmatrix_coo(mat):
        mat = mat.tocoo()
    idxs = np.vstack((mat.row, mat.col)).transpose()
    values = mat.data
    shape = mat.shape
    return idxs, values, shape

#trasforma matrici sparse in tuble
def sparse_to_tuple(sparse_mat):
    if isinstance(sparse_mat, list):
        for i in range(len(sparse_mat)):
            sparse_mat[i] = to_tuple(sparse_mat[i])
    else:
        sparse_mat = to_tuple(sparse_mat)
    return sparse_mat

#normalizza la matrice delle feature 
def process_features(features):
    features /= features.sum(1).reshape(-1, 1)
    features[np.isnan(features) | np.isinf(features)] = 0 
    return features

#renormalization trick della matrice di adiacenza
def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = tf.pow(tf.reduce_sum(adj,axis=1), -0.5)
        d = tf.diag(d)
        a_norm = tf.matmul(tf.matmul(d,adj),d)
    else:
        d = tf.pow(tf.reduce_sum(adj,axis=1), -1)
        d = tf.diag(d)
        a_norm = tf.matmul(d, adj)
    return a_norm


def preprocess_adj(adj, is_gcn, symmetric = True):
    if is_gcn:
        adj = adj + FLAGS.mu * tf.eye(adj.get_shape().as_list()[0]) 
    adj = normalize_adj(adj, symmetric)
    return adj

def normalize_adj_numpy(adj, symmetric=True):
    if symmetric:
        d = np.power(np.sum(adj,axis=1), -0.5)
        d = np.diag(d)
        a_norm = np.matmul(np.matmul(d,adj),d)
    else:
        d = np.power(np.sum(adj,axis=1), -1)
        d = np.diag(d)
        a_norm = np.matmul(d, adj)
    return a_norm

def preprocess_adj_numpy(adj, is_gcn, symmetric=True):
    if is_gcn:
        adj = adj + FLAGS.mu * np.eye(adj.shape[0])
    adj = normalize_adj_numpy(adj, symmetric)
    return adj
#  --------------------- metriche --------------------------------------------
#cross-entropy con mascheramento per isolare i nodi con label
def cross_entropy(predictions, labels):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=labels)
    return tf.reduce_mean(loss)
def binary_cross_entropy(predictions, labels):
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=predictions,
                                                  targets=labels,
                                                  pos_weight=1.)
    return tf.reduce_mean(loss)
#accuracy con mascheramento
def accuracy_cal(predictions, labels):
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    pos_pred = tf.reduce_sum(tf.argmax(predictions,1))
    return tf.reduce_mean(accuracy_all),pos_pred
def binary_accuracy_cal(predictions, labels):
    predict_class = tf.cast(tf.greater(tf.nn.sigmoid(predictions),0.5),tf.float32)
    correct_prediction = tf.equal(predict_class, labels)
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)
#  ----------------------- init -----------------------------------------------
#inizializzatore di pesi secondo Glorot&Bengio
def glorot(shape, name=None):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    val = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(val, name=name)

def zeros(shape, name=None):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

#costruzione del dizionario per GCN 
def build_dictionary_GCN(feats, edge_features, labels,  placeholders):
    #prepara il dizionario che sarà poi passato alla sessione di TF
    dictionary = dict()
    dictionary.update({placeholders['labels']: labels})
    dictionary.update({placeholders['feats']: feats})
    dictionary.update({placeholders['edge_features']: edge_features})
   
    return dictionary
def build_dictionary_GCN_shellnet(feats, edge_features, labels,  coords, placeholders):
    #prepara il dizionario che sarà poi passato alla sessione di TF
    dictionary = dict()
    dictionary.update({placeholders['labels']: labels})
    dictionary.update({placeholders['feats']: feats})
    dictionary.update({placeholders['edge_features']: edge_features})
    #modify here
    dictionary.update({placeholders['points']: coords})
   
    return dictionary

def block_diagonal(matrices):
  r"""Constructs block-diagonal matrices from a list of batched 2D tensors.

  Args:
    matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
      matrices with the same batch dimension).
    dtype: Data type to use. The Tensors in `matrices` must match this dtype.
  Returns:
    A matrix with the input matrices stacked along its main diagonal, having
    shape [..., \sum_i N_i, \sum_i M_i].

  """
  matrices = [tf.convert_to_tensor(matrices[i]) for i in range(matrices.get_shape().as_list()[0])]
  blocked_rows = tf.Dimension(0)
  blocked_cols = tf.Dimension(0)
  for matrix in matrices:
    full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
    blocked_rows += full_matrix_shape[-2]
    blocked_cols += full_matrix_shape[-1]
  ret_columns_list = []
  for matrix in matrices:
    matrix_shape = tf.shape(matrix)
    ret_columns_list.append(matrix_shape[-1])
  ret_columns = tf.add_n(ret_columns_list)
  row_blocks = []
  current_column = 0
  for matrix in matrices:
    matrix_shape = tf.shape(matrix)
    row_before_length = current_column
    current_column += matrix_shape[-1]
    row_after_length = ret_columns - current_column
    row_blocks.append(tf.pad(
        tensor=matrix,
        paddings=tf.concat(
            [tf.zeros([1, 2], dtype=tf.int32),
             [[row_before_length, row_after_length]]],
            axis=0)))
  blocked = tf.concat(row_blocks, axis=0)
  blocked.set_shape((blocked_rows, blocked_cols))
  return blocked