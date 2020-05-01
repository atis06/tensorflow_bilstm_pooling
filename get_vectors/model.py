import tensorflow as tf
import numpy as np


class BiRNNWithPooling:

    def __init__(self, num_inputs, num_time_steps, num_hidden, use_embedding_layer, embedding_matrix_shape):
        # Just one feature, the time series(embeddig dim)
        self.num_inputs = num_inputs
        # Num of steps in each batch (seqlength)
        self.num_time_steps = num_time_steps
        # Hidden neurons in each rnn layer
        self.num_hidden = num_hidden
        # Pooling (max, avg, None)
        self.pooling = 'max'

        self.use_embedding_layer=use_embedding_layer

        self.embedding_matrix_shape = embedding_matrix_shape

        with tf.device('/CPU:0'):
            if self.use_embedding_layer:
                self.saved_embeddings = tf.placeholder(dtype=tf.float32, shape=embedding_matrix_shape)
                self.trained_embedding = tf.get_variable(name='embedding', shape=embedding_matrix_shape, trainable=False, dtype=tf.float32)
                
        self.X = tf.placeholder(tf.int32, [None, self.num_time_steps])

        self.out = self.__get_masked_lm_network()


    def __biRNN(self, input):
        with tf.variable_scope("birnn_pool"):

            fw_cell = tf.nn.rnn_cell.LSTMCell(int(self.num_hidden/2), forget_bias=1.0, dtype=tf.float32, trainable=False)
            bw_cell = tf.nn.rnn_cell.LSTMCell(int(self.num_hidden/2), forget_bias=1.0, dtype=tf.float32, trainable=False)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input,
                                                             dtype=tf.float32)

            output_rnn = tf.concat(outputs, axis=2)  # [batch_size,sequence_length,hidden_size]

            return output_rnn

    def get_output_with_pooling(self, output_rnn):
        if self.pooling is not None:
            if self.pooling == 'max':
                output_rnn_last = tf.reduce_max(output_rnn,axis=1)  # [batch_size,hidden_size]
            if self.pooling == 'avg':
                output_rnn_last = tf.reduce_mean(output_rnn, axis=1)
        else:
            output_rnn_last = output_rnn[:,-1,:]

        return output_rnn_last

    def get_masked_lm_network(self):
        return self.optimizer, self.loss, self.output_logits, self.out

    def __get_masked_lm_network(self):
        embed = tf.nn.embedding_lookup(self.trained_embedding, self.X)
        rnn_output = self.__biRNN(embed)
        rnn_output_pooled = self.get_output_with_pooling(rnn_output)
        return rnn_output_pooled



