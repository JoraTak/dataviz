from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time

import tensorflow as tf
import numpy as np


def matmul3d(X, W):
    """Wrapper for tf.matmul to handle a 3D input tensor X.
    Will perform multiplication along the last dimension.
    Args:
      X: [m,n,k]
      W: [k,l]
    Returns:
      XW: [m,n,l]
    """
    Xr = tf.reshape(X, [-1, tf.shape(X)[2]])
    XWr = tf.matmul(Xr, W)
    newshape = [tf.shape(X)[0], tf.shape(X)[1], tf.shape(W)[1]]
    return tf.reshape(XWr, newshape)


def MakeFancyRNNCell(H, keep_prob, num_layers=1):
    """Make a fancy RNN cell.
    Use tf.nn.rnn_cell functions to construct an LSTM cell.
    Initialize forget_bias=0.0 for better training.
    Args:
      H: hidden state size
      keep_prob: dropout keep prob (same for input and output)
      num_layers: number of cell layers
    Returns:
      (tf.nn.rnn_cell.RNNCell) multi-layer LSTM cell with dropout
    """
    cells = []
    for _ in range(num_layers):
      cell = tf.nn.rnn_cell.BasicLSTMCell(H, forget_bias=0.0)
      cell = tf.nn.rnn_cell.DropoutWrapper(
          cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
      cells.append(cell)
    return tf.nn.rnn_cell.MultiRNNCell(cells)


# Decorator-foo to avoid indentation hell.
# Decorating a function as:
# @with_self_graph
# def foo(self, ...):
#     # do tensorflow stuff
#
# Makes it behave as if it were written:
# def foo(self, ...):
#     with self.graph.as_default():
#         # do tensorflow stuff
#
# We hope this will save you some indentation, and make things a bit less
# error-prone.
def with_self_graph(function):
    def wrapper(self, *args, **kwargs):
        with self.graph.as_default():
            return function(self, *args, **kwargs)
    return wrapper


class RNNLM(object):
    def __init__(self, graph=None, *args, **kwargs):
        """Init function.
        This function just stores hyperparameters. You'll do all the real graph
        construction in the Build*Graph() functions below.
        Args:
          V: vocabulary size
          H: hidden state dimension
          num_layers: number of RNN layers (see tf.nn.rnn_cell.MultiRNNCell)
        """
        # Set TensorFlow graph. All TF code will work on this graph.
        self.graph = graph or tf.Graph()
        self.SetParams(*args, **kwargs)

    @with_self_graph
    def SetParams(self, V, H, softmax_ns=200, num_layers=1,num_classes=2):
        # Model structure; these need to be fixed for a given model.
        self.V = V
        self.H = H
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Training hyperparameters; these can be changed with feed_dict,
        # and you may want to do so during training.
        with tf.name_scope("Training_Parameters"):
            # Number of samples for sampled softmax.
            self.softmax_ns = softmax_ns

            self.learning_rate_ = tf.placeholder(tf.float32, [], name="learning_rate")

            # For gradient clipping, if you use it.
            # Due to a bug in TensorFlow, this needs to be an ordinary python
            # constant instead of a tf.constant.
            self.max_grad_norm_ = 1.0

            self.use_dropout_ = tf.placeholder_with_default(
                False, [], name="use_dropout")

            # If use_dropout is fed as 'True', this will have value 0.5.
            self.dropout_keep_prob_ = tf.cond(
                self.use_dropout_,
                lambda: tf.constant(0.5),
                lambda: tf.constant(1.0),
                name="dropout_keep_prob")

            # Dummy for use later.
            self.no_op_ = tf.no_op()


    @with_self_graph
    def BuildCoreGraph(self):
        """Construct the core LSTM graph.
        """
        # Input ids, with dynamic shape depending on input.
        # Should be shape [batch_size, max_time] and contain integer word indices.
        self.input_w_ = tf.placeholder(tf.int32, [None, None], name="w")

        # Initial hidden state. You'll need to overwrite this with cell.zero_state
        # once you construct your RNN cell.
        self.initial_h_ = None

        # Final hidden state. You'll need to overwrite this with the output from
        # tf.nn.dynamic_rnn so that you can pass it in to the next batch (if
        # applicable).
        self.final_h_ = None

        # Output logits, which can be used by loss functions or for prediction.
        # Overwrite this with an actual Tensor of shape
        # [batch_size, max_time, V].
        self.logits_ = None

        # Should be the same shape as inputs_w_
        self.target_y_ = tf.placeholder(tf.int32, [None], name="y")

        # Replace this with an actual loss function
        self.loss_ = None

        # Get dynamic shape info from inputs
        with tf.name_scope("batch_size"):
            self.batch_size_ = tf.shape(self.input_w_)[0]
        with tf.name_scope("max_time"):
            self.max_time_ = tf.shape(self.input_w_)[1]

        self.ns_ = tf.tile([self.max_time_], [self.batch_size_, ], name="ns")

        # Construct embedding layer
        with tf.name_scope("Embedding_Layer"):
            self.W_in_ = tf.Variable(tf.random_uniform([self.V, self.H], -1.0, 1.0), name="C")            
            self.embed_ = tf.nn.embedding_lookup(self.W_in_, self.input_w_)
                 

        # Construct RNN/LSTM cell and recurrent layer.
        with tf.name_scope("recurrent_Layer"):            
            self.cell_ = MakeFancyRNNCell(self.H, self.dropout_keep_prob_, self.num_layers)
            self.initial_h_ = self.cell_.zero_state(self.batch_size_, tf.float32)                              
            self.outputs_, self.final_h_ = tf.nn.dynamic_rnn(self.cell_, self.embed_,initial_state=self.initial_h_) 
        """
        # Output Layer    
        with tf.name_scope("output_Layer"):
            self.W_out_ = tf.Variable(tf.random_uniform([self.H, self.num_classes], -1.0, 1.0), name="W_out")
            self.b_out_ = tf.Variable(tf.zeros([self.num_classes], dtype=tf.float32), name="b_out")
            self.logits_ = tf.add(matmul3d(self.outputs_, self.W_out_), self.b_out_, name="logits")

        # Loss computation (true loss, for prediction)
        with tf.name_scope("true_loss"):
            self.loss_ = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.target_y_
                    ,logits=tf.reshape(self.logits_,[self.batch_size_,-1])))
            
        """

    @with_self_graph
    def BuildTrainGraph(self):
        """Construct the training ops.
        """
           
            
        with tf.name_scope('predictions'):
            self.predictions_ = tf.contrib.layers.fully_connected(self.outputs_[:, -1], 1, activation_fn=tf.sigmoid)

#        with tf.name_scope('predictions'):
#            self.predictions_ = tf.contrib.layers.fully_connected(self.outputs_[:, -1], 1, #activation_fn=tf.sigmoid,trainable=True)

        with tf.name_scope('train_loss_'):
            self.train_loss_ = tf.losses.mean_squared_error(
                tf.reshape(self.target_y_,(1,-1)), tf.reshape(self.predictions_,(1,-1)))    
            tf.summary.scalar('train_loss_', self.train_loss_)
                  
        # Define optimizer and training op
        self.optimizer_ = tf.train.AdamOptimizer(learning_rate=self.learning_rate_).minimize(self.train_loss_)
        
        with tf.name_scope('accuracy'): 
            predictions = tf.cast(tf.round(self.predictions_), tf.int32)
            self.num_equal_ = tf.equal(predictions, tf.reshape(self.target_y_,(-1,1)))
            self.accuracy_ = tf.reduce_mean(tf.cast(self.num_equal_, tf.float32))

       
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

