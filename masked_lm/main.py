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
num_inputs = 1
num_time_steps = 200
num_hidden = 30
learning_rate = 0.1
batch_size = 100
num_classes = 5
dropout_keep_prob = 1
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



    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(articles)
    word_index = tokenizer.word_index


    sequences = tokenizer.texts_to_sequences(articles)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)

    reverse_word_index = dict([(value , key) for (key, value) in word_index.items()])

    return articles, sequences, tokenizer



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


def next_batch(batch_size, articles, tokenizer, w2v_model):
    idx = np.arange(0, len(articles))
    np.random.shuffle(idx)
    idx = idx[0:batch_size]
    data_shuffle = [articles[i] for i in idx]

    input_examples=[]
    for i, article in enumerate(data_shuffle):
        input_examples.append(features.InputExample(i, articles[i], None))

    features_list = features.convert_examples_to_features(input_examples, 15, tokenizer)

    masked_lm_preds = []
    input_ids = []
    for feature in features_list:
        tokens, masked_lm_positions, masked_lm_labels = utils.create_masked_lm_predictions(feature.tokens, 0.4, 3, w2v_model, random)
        instance = utils.TrainingInstance(
            tokens=tokens,
            segment_ids=feature.input_type_ids,
            is_random_next=True,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        masked_lm_preds.append(instance)
        input_ids.append([id[0] if id !=[] else 1 for id in feature.input_ids])

    return np.asarray(masked_lm_preds),input_ids


def print_shape(varname, var):
    with tf.Session() as sess:
        print('{0} : {1}'.format(varname, tf.shape(var, name=None)))


def train_model():
    articles, sequences, tokenizer = preprocess_data()

    w2v_model = train_w2v(articles)

    embedding_matrix = get_embedding_matrix(w2v_model)

    tokens=[]
    masked_lm_ids = []
    masked_lm_positions = []
    masked_lm_weights = []
    data_batch, input_ids = next_batch(batch_size, articles, tokenizer, w2v_model)

    for data in data_batch:
        tokens.append(np.array(data.tokens))
        ids = tokenizer.texts_to_sequences(data.masked_lm_labels)
        masked_lm_ids.append([masked_lm_id[0] for masked_lm_id in ids if masked_lm_id != []])
        masked_lm_weights.append([1.0] * len(ids))
        masked_lm_positions.append(data.masked_lm_positions)

    tokens=np.asarray(tokens)

    input_ids = [np.asarray(ids) - 1  for ids in input_ids]

    masked_lm_ids = np.asarray(masked_lm_ids)
    masked_lm_positions = np.asarray(masked_lm_positions)
    masked_lm_weights = np.asarray(masked_lm_weights)

    seqlen = len(tokens[0])

    '''print(tokens[0])
    print(input_ids[0])
    # print('orig: ' + str([reverse_word_index[word+1] for word in masked_lm_ids[0]]))
    print(masked_lm_positions[0])
    print(masked_lm_weights[0])

    print('ORIGS')
    for i in masked_lm_ids:
        print(str([w2v_model.wv.index2word[word] for word in i]))
    print('END')

    #print("to predict: " + str([reverse_word_index[word] for word in masked_lm_ids]))'''

    tokens = np.asarray(['[CLS]', 'tv', 'future', '[MASK]', 'viewers', 'home', 'theatre', '[MASK]',
     'plasma','high-definition', 'tvs', 'digital', '[MASK]', 'recorders', '[SEP]'])

    print(tokens)
    input_ids = []
    input_ids.append([w2v_model.wv.vocab.get(token).index if w2v_model.wv.vocab.get(token) is not None else 0 for token in tokens])
    input_ids = np.asarray(input_ids)
    print(input_ids)
    masked_lm_positions = np.asarray([3, 7, 12])
    print(masked_lm_positions)
    masked_lm_weights = [1.]*len(input_ids)
    print(masked_lm_weights)
    masked_lm_ids = np.asarray([w2v_model.wv.vocab.get('hands').index, w2v_model.wv.vocab.get('systems').index, w2v_model.wv.vocab.get('video').index])
    print(masked_lm_ids)




    model = BiRNNWithPooling(1, seqlen, num_hidden, learning_rate, num_classes, dropout_keep_prob, pooling, use_embedding_layer, embedding_matrix)

    init = tf.global_variables_initializer()


    with tf.Session() as sess:
        sess.run(init)
        feed_dict = {model.X: input_ids, model.positions: masked_lm_positions, model.label_ids: masked_lm_ids,
                     model.label_weights: masked_lm_weights}
        for i in range(100000):

            #print(sess.run(model.out, feed_dict).shape)
            out = np.argmax(sess.run(model.out, feed_dict), axis=-1)
            for o in out:
                print(w2v_model.wv.index2word[o])
            print('###')
                #print((sess.run(model.out, feed_dict)[1]))
            model.train_masked_lm(sess, feed_dict)

        result = sess.run(model.out, feed_dict)
        #print(result.shape)
        #print('res: ' + str(np.argmax(result, axis=-1)))
        #print('prediction: ' + str([reverse_word_index[word] for word in np.argmax(result, axis=-1)]))

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
