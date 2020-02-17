import csv
import tensorflow as tf
import numpy as np
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import gensim

from model import BiRNNWithPooling

print(tf.__version__)

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Preprocess parameters
vocab_size = 10000
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = 'OOV'
training_portion = .8

# W2V parameters
w2v_dim = 300
min_count = 2
iter = 5
epochs = 15

## Network training parameters

net_epochs = 150
num_inputs = 300
num_time_steps = 200
num_hidden = 200
learning_rate = 0.001
batch_size = 100
num_classes = 5
dropout_keep_prob = 0.8
pooling = 'max'

use_embedding_layer=True

def preprocess_data():
    articles = []
    labels = []

    with open("bbc-text.csv", 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            labels.append(row[0])
            article = row[1]
            for word in STOPWORDS:
                token = ' ' + word + ' '
                article = article.replace(token, ' ')
                article = article.replace(' ', ' ')
            articles.append(article)

    train_size = int(len(articles) * training_portion)
    train_articles = articles[0: train_size]
    train_labels = labels[0: train_size]

    validation_articles = articles[train_size:]
    validation_labels = labels[train_size:]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_articles)
    word_index = tokenizer.word_index


    train_sequences = tokenizer.texts_to_sequences(train_articles)
    train_padded = np.asarray(pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type), dtype='int32')

    validation_sequences = tokenizer.texts_to_sequences(validation_articles)
    validation_padded = np.asarray(pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type), dtype='int32')

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
    validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

    return train_padded, train_labels, validation_padded, validation_labels, reverse_word_index, articles, training_label_seq, validation_label_seq


def decode_article(text, reverse_word_index):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


def train_w2v(articles):
    splitted_articles = [article.split() for article in articles]
    model = gensim.models.Word2Vec(iter=iter, size=w2v_dim, workers=4, min_count=min_count)
    model.build_vocab(splitted_articles)
    model.train(splitted_articles, total_examples=len(splitted_articles), epochs=epochs)

    return model

def get_embedding_matrix(w2v_model):
    embedding_matrix = np.zeros((len(w2v_model.wv.vocab), w2v_dim))
    for i in range(len(w2v_model.wv.vocab)):
        embedding_vector = w2v_model.wv[w2v_model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def apply_w2v(model, train_padded, validation_padded, reverse_word_index):
    train_articles_w2v = []
    validation_articles_w2v = []

    for row in train_padded:
        w2v_sentence = []
        for word in enumerate(decode_article(row, reverse_word_index).split()):
            try:
                w2v_sentence.append(np.asarray(model.wv[word[1]]))
            except Exception as e:
                w2v_sentence.append(np.asarray(np.zeros(w2v_dim)))
        train_articles_w2v.append(np.asarray(w2v_sentence))

    for row in validation_padded:
        w2v_sentence = []
        for word in enumerate(decode_article(row, reverse_word_index).split()):
            try:
                w2v_sentence.append(np.asarray(model.wv[word[1]]))
            except Exception as e:
                w2v_sentence.append(np.asarray(np.zeros(w2v_dim)))
        validation_articles_w2v.append(np.asarray(w2v_sentence))

    return train_articles_w2v, validation_articles_w2v


def next_batch(batch_size, train_articles_w2v, training_label_seq):
    idx = np.arange(0, len(train_articles_w2v))
    np.random.shuffle(idx)
    idx = idx[0:batch_size]
    data_shuffle = [train_articles_w2v[i] for i in idx]
    labels_shuffle = [training_label_seq[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def print_shape(varname, var):
    with tf.Session() as sess:
        print('{0} : {1}'.format(varname, tf.shape(var, name=None)))


def train_model():
    train_padded, train_labels, validation_padded, validation_labels, reverse_word_index, articles, training_label_seq, validation_label_seq = preprocess_data()
    w2v_model = train_w2v(articles)
    train_articles_w2v, validation_articles_w2v = apply_w2v(w2v_model, train_padded, validation_padded,
                                                            reverse_word_index)

    embedding_matrix = get_embedding_matrix(w2v_model)

    model = BiRNNWithPooling(num_inputs, num_time_steps, num_hidden, learning_rate, num_classes, dropout_keep_prob, pooling, True, embedding_matrix)
    init = tf.global_variables_initializer()


    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(net_epochs):
            X_batch, y_batch = next_batch(batch_size, train_padded, training_label_seq)

            if use_embedding_layer:
                X_batch = X_batch.reshape((batch_size, num_time_steps))
            else:
                X_batch = X_batch.reshape((batch_size, num_time_steps, num_inputs))
            y_batch = tf.one_hot(y_batch - 1, num_classes, axis=-1).eval().reshape(-1, num_classes)

            feed_dict = {model.X: X_batch, model.y: y_batch}
            model.train(sess, feed_dict)

            if epoch % 100 == 0:
                feed_dict = {model.X: X_batch, model.y: y_batch}
                mse = model.evaluate(feed_dict)
                print(epoch, "\tMSE:", mse)


        if use_embedding_layer:
            X_batch_valid = np.asarray(validation_padded).reshape((len(validation_articles_w2v), num_time_steps))
        else:
            X_batch_valid = np.asarray(validation_padded).reshape((len(validation_articles_w2v), num_time_steps, num_inputs))
        y_batch_valid = tf.one_hot(validation_label_seq - 1, num_classes, axis=-1).eval().reshape(-1, num_classes)

        feed_dict_valid = {model.X: X_batch_valid, model.y: y_batch_valid}

        loss_valid, acc_valid = model.validate(sess, feed_dict_valid)
        print('---------------------------------------------------------')
        print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
              format(epoch + 1, loss_valid, acc_valid))
        print('---------------------------------------------------------')


        if use_embedding_layer:
            X_batch = X_batch.reshape((1, num_time_steps, num_inputs))
        else:
            X_batch, y_batch = next_batch(1, train_padded, training_label_seq)
        X_batch = X_batch.reshape((1, num_time_steps))
        y_batch = tf.one_hot(y_batch - 1, num_classes, axis=-1).eval().reshape(-1, num_classes)

        feed_dict = {model.X: X_batch, model.y: y_batch}
        print(np.argmax(sess.run(model.y, feed_dict)) == np.argmax(y_batch))


train_model()
