import tensorflow as tf
import utils
import numpy as np


class BiRNNWithPooling:

    def __init__(self, num_inputs, num_time_steps, num_hidden, learning_rate, num_classes, dropout_keep_prob, pooling, use_embedding_layer, embedding_matrix):
        # Just one feature, the time series(embeddig dim)
        self.num_inputs = num_inputs
        # Num of steps in each batch (seqlength)
        self.num_time_steps = num_time_steps
        # Hidden neurons in each rnn layer
        self.num_hidden = num_hidden
        # Learning rate you can play with this
        self.learning_rate = learning_rate
        # Num of classes
        self.num_classes = num_classes
        # Dropout keep probability (can be None)
        self.dropout_keep_prob = dropout_keep_prob
        # Pooling (max, avg, None)
        self.pooling = pooling

        self.use_embedding_layer=use_embedding_layer

        self.embedding_matrix = embedding_matrix

        if self.use_embedding_layer:
            self.saved_embeddings = tf.constant(embedding_matrix, dtype=tf.float32)
            self.embedding = tf.Variable(initial_value=self.saved_embeddings, trainable=False)

        #self.X, self.y = self.__init_placeholders()
        self.X = tf.placeholder(tf.int32, [None, self.num_time_steps])
        self.W, self.b = self.__init_variables()

        self.positions, self.label_ids, self.label_weights = self.__init_placeholders_masked_lm()

        self.optimizer, self.loss, self.output_logits, self.out = self.__get_masked_lm_network()




    def __init_placeholders(self):
        if self.use_embedding_layer:
            X = tf.placeholder(tf.int32, [None, self.num_time_steps])
        else:
            X = tf.placeholder(tf.float32, [None, self.num_time_steps, self.num_inputs])
        y = tf.placeholder(tf.float32, [None, self.num_classes])

        return X, y

    def __init_placeholders_masked_lm(self):
        positions = tf.placeholder(tf.int32)
        label_ids = tf.placeholder(tf.int32)
        label_weights = tf.placeholder(tf.float64)

        return positions, label_ids, label_weights

    def __init_variables(self):
        W = tf.Variable(tf.random_normal(shape=[2 * self.num_hidden, self.num_classes]), dtype=tf.float32)
        b = tf.Variable(tf.ones([self.num_classes], dtype=tf.float32))

        return W, b

    def __biRNN(self, input, full_output = False):
        with tf.variable_scope("birnn_pool"):

            fw_cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden, forget_bias=1.0)
            bw_cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden, forget_bias=1.0)

            if self.dropout_keep_prob is not None:
                    fw_cell=tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=self.dropout_keep_prob)
                    bw_cell=tf.contrib.rnn.DropoutWrapper(bw_cell,output_keep_prob=self.dropout_keep_prob)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input,
                                                         dtype=tf.float32)

            output_rnn = tf.concat(outputs, axis=2)  # [batch_size,sequence_length,hidden_size*2]

        if full_output:
            return output_rnn
        else:
            # this is pooling
            if self.pooling is not None:
                if self.pooling == 'max':
                    output_rnn_last = tf.reduce_max(output_rnn,axis=1)  # [batch_size,hidden_size*2]
                if self.pooling == 'avg':
                    output_rnn_last = tf.reduce_mean(output_rnn, axis=1)
            else:
                # this uses the last hidden state as the representation.
                # [batch_size,hidden_size*2]
                output_rnn_last = output_rnn[:,-1,:]


            return output_rnn_last

    def __dense_layer(self, input, weights, biases):
        output_logits = tf.matmul(input, weights) + biases
        #y_pred = tf.nn.softmax(output_logits)

        return output_logits

    def print_shape(self, varname, var):
        with tf.Session() as sess:
            print('{0} : {1}'.format(varname, tf.shape(var, name=None)))

    def __loss(self, output_logits, y):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=output_logits))
        return loss

    def __accuracy(self, output_logits, y):
        correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy

    def __optimizer(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)

        return optimizer

    '''def __get_network(self):
        if self.use_embedding_layer:
            self.embed = tf.nn.embedding_lookup(self.embedding, self.X)
            #embed=tf.unstack(embed)
            rnn_output = self.__biRNN(self.embed)
        else:
            rnn_output = self.__biRNN(self.X)
        output_logits = tf.layers.dense(rnn_output, units=self.num_classes)
        #output_logits = self.__dense_layer(rnn_output, self.W, self.b)
        loss = self.__loss(output_logits, self.y)
        optimizer = self.__optimizer(loss)

        return optimizer, loss, output_logits

    def get_network(self):
        return self.optimizer, self.loss, self.output_logits'''

    def train(self, sess, feed_dict):
        optimizer, loss, output_logits = self.get_network()
        sess.run(optimizer, feed_dict)

    def evaluate(self, feed_dict):
        optimizer, loss, output_logits = self.get_network()
        mse = loss.eval(feed_dict)

        return mse

    def validate(self, sess, feed_dict_valid):
        optimizer, loss, output_logits = self.get_network()
        accuracy = self.__accuracy(output_logits, self.y)
        loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)

        return loss_valid, acc_valid

    def gelu(self, x):
        """Gaussian Error Linear Unit.

        This is a smoother version of the RELU.
        Original paper: https://arxiv.org/abs/1606.08415
        Args:
          x: float Tensor to perform activation.

        Returns:
          `x` with the GELU activation applied.
        """
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
        self.embed = tf.nn.embedding_lookup(self.embedding, self.X)
        rnn_output = self.__biRNN(self.embed, True)

        # MASKED LM
        input_tensor = utils.gather_indexes(rnn_output, self.positions)

        input_tensor = tf.layers.dense(input_tensor, units=self.num_hidden*2, activation=self.gelu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

        input_tensor = utils.layer_norm(input_tensor)
        input_tensor = tf.cast(input_tensor, tf.float64)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(name='output_bias',
            shape=[vocab_size],
            initializer=tf.zeros_initializer(), dtype=tf.float64)
        logits = tf.matmul(input_tensor, self.embedding_matrix, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        #log_probs=tf.nn.softmax(logits)

        label_ids = tf.reshape(self.label_ids, [-1])
        label_weights = tf.reshape(self.label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=vocab_size, dtype=tf.float64)


        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator
        optimizer = self.__optimizer(loss)

        out = log_probs

        return optimizer, loss, logits, out

    def train_masked_lm(self, sess, feed_dict):
        loss, optimizer, logits, out = self.get_masked_lm_network()
        sess.run(optimizer, feed_dict)


