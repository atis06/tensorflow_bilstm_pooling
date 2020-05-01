from os import listdir
from os.path import isfile, join
from gensim.models import Word2Vec
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import collections
import os
import sys
import math
import os

import functools

print = functools.partial(print, flush=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# nltk.download('punkt')
from itertools import islice, chain, tee
import random
import tensorflow as tf
# tf.disable_v2_behavior()
from model import BiRNNWithPooling
from tensorflow.python.client import device_lib


config = tf.ConfigProto(
    device_count={'GPU': 2}
)

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

# Config
save_model_to = '../masked_lm/model/mlm_model.ckpt'

w2v_model = Word2Vec.load(
    '/home/eattknt/git/thesis/tensorflow_bilstm_pooling/masked_lm/model/enc-hu-oscar-hun-spacy.w2v')
w2v_dim = 300

tokens_path = '../masked_lm/asd'

# network config
num_inputs = 1

num_hidden = 1024
use_embedding_layer = True

batch_size = 1


num_time_steps = 2 * 100 + 1
w2v_vocab_len = len(w2v_model.wv.vocab)


def get_embedding_matrix(w2v_model):
    embedding_matrix = np.zeros((len(w2v_model.wv.vocab) + 4, w2v_dim + 4))
    for i in range(len(w2v_model.wv.vocab)):
        embedding_vector = np.append(w2v_model.wv[w2v_model.wv.index2word[i]], np.zeros(4))
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_matrix[len(w2v_model.wv.vocab)] = np.append(np.zeros(w2v_dim), [1, 0, 0, 0]) # [PAD]
    embedding_matrix[len(w2v_model.wv.vocab) + 1] = np.append(np.zeros(w2v_dim), [0, 1, 0, 0]) # [MASK]
    embedding_matrix[len(w2v_model.wv.vocab) + 2] = np.append(np.zeros(w2v_dim), [0, 0, 1, 0]) # [SEP]
    embedding_matrix[len(w2v_model.wv.vocab) + 3] = np.append(np.zeros(w2v_dim), [0, 0, 0, 1]) # unknown

    return embedding_matrix


embedding_matrix = get_embedding_matrix(w2v_model)

def extract_data(data):
    input_ids = []

    for elem in data:
        input_ids.append(elem)

    return np.asarray(input_ids)

def chunks(iterable, rnd=False):
    iterator = iter(iterable)
    for first in iterator:
        if rnd:
            std = sorted(list(chain([first], islice(iterator, batch_size - 1))), key=lambda k: random.random())
            yield std
        else:
            yield list(chain([first], islice(iterator, batch_size - 1)))


def load_data():
    input_ids_file = open("../masked_lm/data/input_ids", "r")

    for input_ids in input_ids_file:
        yield np.asarray(input_ids.split()).astype(int)


model = BiRNNWithPooling(num_inputs, num_time_steps, num_hidden, use_embedding_layer, embedding_matrix.shape)

init = tf.global_variables_initializer()

saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "birnn"))

data_gen = load_data()

with tf.Session(config=config) as sess:
    sess.run(init)
    if os.path.isfile('../masked_lm/model/5/mlm-model-epoch5.ckpt.index'):
        print('Restoring weights...')
        saver.restore(sess, '../masked_lm/model/5/mlm-model-epoch5.ckpt')
        print('Weights restored...')
    sess.run(model.trained_embedding.assign(model.saved_embeddings), {model.saved_embeddings: embedding_matrix})
    for i, data in enumerate(chunks(data_gen)):
        data = np.asarray(data).reshape(-1, num_time_steps)
        input_ids = extract_data(data)

        feed_dict = {model.X: input_ids}

        out = sess.run(model.out, feed_dict)
        print(out[0])
