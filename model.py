import tensorflow as tf

class BiRNNWithPooling:

    def __init__(self):
        # Just one feature, the time series(embeddig dim)
        self.num_inputs = 300
        # Num of steps in each batch (seqlength)
        self.num_time_steps = 200
        # Hidden neurons in each rnn layer
        self.num_hidden = 200
        # Learning rate you can play with this
        self.learning_rate = 0.001
        # How many iterations to go through (training steps)
        self.epochs = 200
        # Size of the batch of data
        self.batch_size = 100
        # Num of classes
        self.n_classes = 5
        # Dropout keep probability
        self.dropout_keep_prob = 0.8
        # Pooling (max, avg, None)
        self.pooling = 'max'

        self.X, self.y = self.__init_placeholders()

        self.W, self.b = self.__init_variables()

        self.optimizer, self.loss, self.output_logits = self.__get_network()

    def __init_placeholders(self):
        X = tf.placeholder(tf.float32, [None, self.num_time_steps, self.num_inputs])
        y = tf.placeholder(tf.float32, [None, self.n_classes])

        return X, y

    def __init_variables(self):
        W = tf.Variable(tf.random_normal(shape=[2 * self.num_hidden, self.n_classes]))
        b = tf.Variable(tf.zeros([self.n_classes]))

        return W, b

    def __biRNN(self, input):
        fw_cell = tf.contrib.rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0)
        bw_cell = tf.contrib.rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0)

        if self.dropout_keep_prob is not None:
                fw_cell=tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=self.dropout_keep_prob)
                bw_cell=tf.contrib.rnn.DropoutWrapper(bw_cell,output_keep_prob=self.dropout_keep_prob)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input,
                                                     dtype=tf.float32)

        output_rnn = tf.concat(outputs, axis=2)  # [batch_size,sequence_length,hidden_size*2]

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


    def __loss(self, output_logits, y):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits))
        return loss

    def __accuracy(self, output_logits, y):
        correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy

    def __optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        return optimizer

    def __get_network(self):
        rnn_output = self.__biRNN(self.X)
        output_logits = self.__dense_layer(rnn_output, self.W, self.b)
        loss = self.__loss(output_logits, self.y)
        optimizer = self.__optimizer(loss)

        return optimizer, loss, output_logits

    def get_network(self):
        return self.optimizer, self.loss, self.output_logits

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
