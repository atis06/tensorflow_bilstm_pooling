import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import utils
import numpy as np


class BiRNNWithPooling:

    def __init__(self, num_inputs, num_time_steps, num_hidden, learning_rate, dropout_keep_prob, pooling, use_embedding_layer, embedding_matrix):
        # Just one feature, the time series(embeddig dim)
        self.num_inputs = num_inputs
        # Num of steps in each batch (seqlength)
        self.num_time_steps = num_time_steps
        # Hidden neurons in each rnn layer
        self.num_hidden = num_hidden
        # Learning rate you can play with this
        self.learning_rate = learning_rate
        # Dropout keep probability (can be None)
        self.dropout_keep_prob = dropout_keep_prob
        # Pooling (max, avg, None)
        self.pooling = pooling

        self.use_embedding_layer=use_embedding_layer

        self.embedding_matrix = embedding_matrix

        '''if self.use_embedding_layer:
            self.saved_embeddings = tf.constant(embedding_matrix, dtype=tf.float32)
            self.embedding = tf.Variable(initial_value=self.saved_embeddings, trainable=False, dtype=tf.float32)'''

        with tf.device('/GPU:1'):
            if self.use_embedding_layer:
                self.saved_embeddings = tf.placeholder(dtype=embedding_matrix.dtype, shape=[embedding_matrix.shape[0], embedding_matrix.shape[1]])
                self.trained_embedding = tf.get_variable(name='embedding', shape=[embedding_matrix.shape[0], embedding_matrix.shape[1]], trainable=False, dtype=tf.float64)
                unk_embedding = tf.get_variable(name="unk_embedding", shape=[1, embedding_matrix.shape[1]], initializer=tf.zeros_initializer, trainable=False, dtype=tf.float64)
                self.embedding = tf.concat([self.trained_embedding, unk_embedding], axis=0)

        self.X = tf.placeholder(tf.int32, [None, self.num_time_steps])

        self.positions, self.label_ids, self.label_weights = self.__init_placeholders_masked_lm()
        self.sentence_labels = self.__init_placeholders_next_sentence()

        self.optimizer, self.loss, self.output_logits, self.out = self.__get_masked_lm_network()

    def __init_placeholders_masked_lm(self):
        positions = tf.placeholder(tf.int32)
        label_ids = tf.placeholder(tf.int32)
        label_weights = tf.placeholder(tf.float64)

        return positions, label_ids, label_weights

    def __init_placeholders_next_sentence(self):
        sentence_labels = tf.placeholder(tf.int32)

        return sentence_labels

    def __biRNN(self, input, full_output = False):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            with tf.variable_scope("birnn_pool"):

                fw_cell = tf.nn.rnn_cell.LSTMCell(int(self.num_hidden/2), forget_bias=1.0)
                bw_cell = tf.nn.rnn_cell.LSTMCell(int(self.num_hidden/2), forget_bias=1.0)

                if self.dropout_keep_prob is not None:
                        fw_cell=tf.nn.rnn_cell.DropoutWrapper(fw_cell,output_keep_prob=self.dropout_keep_prob)
                        bw_cell=tf.nn.rnn_cell.DropoutWrapper(bw_cell,output_keep_prob=self.dropout_keep_prob)

                outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input,
                                                             dtype=tf.float64)

                output_rnn = tf.concat(outputs, axis=2)  # [batch_size,sequence_length,hidden_size]

            return output_rnn

    def get_output_with_pooling(self, output_rnn):
        # this is pooling
        if self.pooling is not None:
            if self.pooling == 'max':
                output_rnn_last = tf.reduce_max(output_rnn,axis=1)  # [batch_size,hidden_size]
            if self.pooling == 'avg':
                output_rnn_last = tf.reduce_mean(output_rnn, axis=1)
        else:
            # this uses the last hidden state as the representation.
            # [batch_size,hidden_size]
            output_rnn_last = output_rnn[:,-1,:]

        return output_rnn_last


    def print_shape(self, varname, var):
        with tf.Session() as sess:
            print('{0} : {1}'.format(varname, tf.shape(var, name=None)))

    def __optimizer(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)

        return optimizer

    def gelu(self, x):
        cdf = 0.5 * (1.0 + tf.tanh(
            (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
        return x * cdf

    def get_masked_lm_network(self):
        return self.optimizer, self.loss, self.output_logits, self.out

    def __get_masked_lm_network(self):
        """Get loss and log probs for the masked LM."""
            # INIT
            vocab_size = self.embedding_matrix.shape[0]
            embedding_size = self.embedding_matrix.shape[1]


            # RNN
            with tf.device('/CPU:0'):
                self.embed = tf.nn.embedding_lookup(self.embedding, self.X)
            rnn_output = self.__biRNN(self.embed, True)
            rnn_output_pooled = self.get_output_with_pooling(rnn_output)


            # MASKED LM
            input_tensor = utils.gather_indexes(rnn_output, self.positions)

            input_tensor = tf.layers.dense(input_tensor, units=embedding_size, activation=self.gelu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

            input_tensor = utils.layer_norm(input_tensor)

            input_tensor = tf.cast(input_tensor, tf.float64)

            # The output weights are the same as the input embeddings, but there is
            # an output-only bias for each token.
            output_bias = tf.get_variable(name='output_bias',
                shape=[vocab_size],
                initializer=tf.zeros_initializer(), dtype=tf.float64)
            logits = tf.matmul(input_tensor, self.trained_embedding, transpose_b=True) # * [vocab_size, embedding_size]
            logits = tf.nn.bias_add(logits, output_bias)

            log_probs = tf.nn.log_softmax(logits, axis=-1)

            log_probs_mlm = tf.exp(log_probs)

            label_ids = tf.reshape(self.label_ids, [-1])
            label_weights = tf.reshape(self.label_weights, [-1])

            one_hot_labels = tf.one_hot(
                label_ids, depth=vocab_size, dtype=tf.float64)

            per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
            numerator = tf.reduce_sum(label_weights * per_example_loss)
            denominator = tf.reduce_sum(label_weights) + 1e-5
            loss_mlm = numerator / denominator


            # NEXT SENTENCE
            output_weights_ns = tf.get_variable(
                "output_weights_ns",
                shape=[2, self.num_hidden],
                initializer=tf.truncated_normal_initializer(stddev=0.02), dtype=tf.float64)
            output_bias_ns = tf.get_variable(
                "output_bias_ns", shape=[2], initializer=tf.zeros_initializer(), dtype=tf.float64)

            logits = tf.matmul(rnn_output_pooled, output_weights_ns, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias_ns)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            log_probs_ns = tf.exp(log_probs)

            labels = tf.reshape(self.sentence_labels, [-1])
            one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float64)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss_next_sentence = tf.reduce_mean(per_example_loss)

            loss = loss_mlm + loss_next_sentence

            optimizer = self.__optimizer(loss)

            out = (loss, loss_mlm, loss_next_sentence, log_probs_mlm, log_probs_ns)

            return optimizer, loss, logits, out

    def train_masked_lm(self, sess, feed_dict):
        sess.run(self.optimizer, feed_dict)


