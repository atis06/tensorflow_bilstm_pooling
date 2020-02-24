import masked_lm
from masked_lm.model import BiRNNWithPooling

import csv
import tensorflow as tf
import numpy as np
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
import gensim
import utils
import features
import random
import sys
np.set_printoptions(threshold=sys.maxsize)



print(tf.__version__)

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
#STOPWORDS = []

# Preprocess parameters
vocab_size = 5000
oov_tok = 'OOV'
training_portion = .8

# W2V parameters
w2v_dim = 300
min_count = 2
iter = 5
epochs = 15

## Network training parameters

net_epochs = 10
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

    with open("masked_lm/bbc-text.csv", 'r') as csvfile:
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



    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(articles)
    word_index = tokenizer.word_index


    sequences = tokenizer.texts_to_sequences(articles)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    return articles, sequences, tokenizer, word_index, reverse_word_index



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
        for word in enumerate(utils.decode_article(row, reverse_word_index).split()):
            try:
                w2v_sentence.append(np.asarray(model.wv[word[1]]))
            except Exception as e:
                w2v_sentence.append(np.asarray(np.zeros(w2v_dim)))
        train_articles_w2v.append(np.asarray(w2v_sentence))

    for row in validation_padded:
        w2v_sentence = []
        for word in enumerate(utils.decode_article(row, reverse_word_index).split()):
            try:
                w2v_sentence.append(np.asarray(model.wv[word[1]]))
            except Exception as e:
                w2v_sentence.append(np.asarray(np.zeros(w2v_dim)))
        validation_articles_w2v.append(np.asarray(w2v_sentence))

    return train_articles_w2v, validation_articles_w2v


def next_batch(batch_size, articles):
    idx = np.arange(0, len(articles))
    np.random.shuffle(idx)
    idx = idx[0:batch_size]
    data_shuffle = [articles[i] for i in idx]

    return np.asarray(data_shuffle)


def print_shape(varname, var):
    with tf.Session() as sess:
        print('{0} : {1}'.format(varname, tf.shape(var, name=None)))


def train_model():
    articles, sequences, tokenizer, word_index, reverse_word_index = preprocess_data()

    input_example = [features.InputExample(1, articles[0][0:200], None)]
    #print(input_example)

    #len(input_example[0].text_a)
    features_list = features.convert_examples_to_features(input_example, 200, tokenizer)
    feature = features_list[0]

    #print(feature.)


    (tokens, masked_lm_positions,
     masked_lm_labels) = utils.create_masked_lm_predictions(
        feature.tokens, 0.2, 50, reverse_word_index, random)
    instance = utils.TrainingInstance(
        tokens=tokens,
        segment_ids=feature.input_type_ids,
        is_random_next=True,
        masked_lm_positions=masked_lm_positions,
        masked_lm_labels=masked_lm_labels)

    print(tokens)

    masked_lm_ids = tokenizer.texts_to_sequences(instance.masked_lm_labels)
    masked_lm_ids = [masked_lm_id[0] for masked_lm_id in masked_lm_ids]
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    print(masked_lm_ids)
    #print(masked_lm_positions)
    #print(masked_lm_weights)

    w2v_model = train_w2v(articles)

    embedding_matrix = get_embedding_matrix(w2v_model)
    #X_batch = next_batch(batch_size, articles)
    #print(X_batch)
    X_batch = np.asarray([index[0] for index in tokenizer.texts_to_sequences(tokens)])
    X_batch = X_batch.reshape(1, len(tokens))
    model = BiRNNWithPooling(1, len(tokens), num_hidden, learning_rate, num_classes, dropout_keep_prob, pooling, use_embedding_layer, embedding_matrix)

    init = tf.global_variables_initializer()


    with tf.Session() as sess:
        sess.run(init)
        feed_dict = {model.X: X_batch, model.positions: masked_lm_positions, model.label_ids: masked_lm_ids,
                     model.label_weights: masked_lm_weights}
        for i in range(1000000):
            model.train_masked_lm(sess, feed_dict)

        result = np.argmax(sess.run(model.out, feed_dict), axis=1)
        print(result)
        #print([reverse_word_index[word] for word in result])

    '''       if epoch % 100 == 0:
                feed_dict = {model.X: X_batch, model.y: y_batch}
                mse = model.evaluate(feed_dict)
                print(epoch, "\tMSE:", mse)

        print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "birnn"))

        if use_embedding_layer:
            X_batch_valid = np.asarray(validation_padded).reshape(len(validation_padded), num_time_steps)
        else:
            X_batch_valid = np.asarray(validation_articles_w2v).reshape(len(validation_articles_w2v), num_time_steps, num_inputs)
        y_batch_valid = tf.one_hot(validation_label_seq - 1, num_classes, axis=-1).eval().reshape(-1, num_classes)

        feed_dict_valid = {model.X: X_batch_valid, model.y: y_batch_valid}

        loss_valid, acc_valid = model.validate(sess, feed_dict_valid)
        print('---------------------------------------------------------')
        print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
              format(epoch + 1, loss_valid, acc_valid))
        print('---------------------------------------------------------')

        if use_embedding_layer:
            X_batch, y_batch = next_batch(1, train_padded, training_label_seq)
            X_batch = X_batch.reshape((1, num_time_steps))
        else:
            X_batch, y_batch = next_batch(1, train_articles_w2v, training_label_seq)
            X_batch = X_batch.reshape((1, num_time_steps, num_inputs))

        y_batch = tf.one_hot(y_batch - 1, num_classes, axis=-1).eval().reshape(-1, num_classes)

        feed_dict = {model.X: X_batch, model.y: y_batch}
        print(np.argmax(sess.run(model.y, feed_dict)) == np.argmax(y_batch))'''


train_model()
