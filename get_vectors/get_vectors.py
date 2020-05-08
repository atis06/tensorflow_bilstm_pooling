from os import listdir
from bse.repo.cassandra_repo import CassandraRepoManager
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
np.set_printoptions(threshold=sys.maxsize)

import functools

print = functools.partial(print, flush=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

repo_manager = CassandraRepoManager()
project_name = 'enc_hu'
data_set_repo_name = 'arukereso_analyzed'
storage_repo_name = 'arukereso_1024'
data_set_repo = repo_manager.get_repo(project_name, data_set_repo_name)
storage_repo = repo_manager.get_repo(project_name, storage_repo_name)

# nltk.download('punkt')
from itertools import islice, chain, tee
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
# tf.disable_v2_behavior()
from model import BiRNNWithPooling
from tensorflow.python.client import device_lib


os.environ['CUDA_VISIBLE_DEVICES'] = ''

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

# Config
save_model_to = '../masked_lm/model/mlm_model.ckpt'

w2v_model = Word2Vec.load(
    '/srv/project/encoder/model/w2v/enc-hu-oscar_sm-hun-spacy/enc-hu-oscar_sm-hun-spacy.w2v')
w2v_dim = 300

# network config
num_inputs = 1

num_hidden = 1024
use_embedding_layer = True

batch_size = 1


num_time_steps = 2 * 100 + 1
vocab_length = len(w2v_model.wv.vocab)

def get_embedding_matrix(w2v_model):
    embedding_matrix = np.zeros((vocab_length + 4, w2v_dim + 4))
    for i in range(len(w2v_model.wv.vocab)):
        embedding_vector = np.append(w2v_model.wv[w2v_model.wv.index2word[i]], np.zeros(4))
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_matrix[vocab_length] = np.append(np.zeros(w2v_dim), [1, 0, 0, 0]) # [PAD]
    embedding_matrix[vocab_length + 1] = np.append(np.zeros(w2v_dim), [0, 1, 0, 0]) # [MASK]
    embedding_matrix[vocab_length + 2] = np.append(np.zeros(w2v_dim), [0, 0, 1, 0]) # [SEP]
    embedding_matrix[vocab_length + 3] = np.append(np.zeros(w2v_dim), [0, 0, 0, 1]) # unknown

    return embedding_matrix


embedding_matrix = get_embedding_matrix(w2v_model)

def extract_data(data):
    input_ids = []

    for elem in data:
        input_ids.append(elem)

    return np.asarray(input_ids)

def chunks(iterable, rnd=False):
    iterator = iter(iterable)
    for id, first in iterator:
        if rnd:
            std = sorted(list(chain([first], islice(iterator, batch_size - 1))), key=lambda k: random.random())
            yield std
        else:
            yield id, list(chain([first], islice(iterator, batch_size - 1)))


def load_data():
    for id, lines in data_set_repo.list():
        tokens = lines.split()
        if len(tokens)>num_time_steps:
            tokens = tokens[0:num_time_steps]
        tokens = tokens + ['[PAD]' for i in range(num_time_steps - len(tokens))]
        
        input_ids = []
        for token in tokens:
            if w2v_model.wv.vocab.get(token) is not None:
                input_ids.append(w2v_model.wv.vocab.get(token).index)
            elif token == '[PAD]':
                input_ids.append(vocab_length)
            elif token == '[MASK]':
                input_ids.append(vocab_length + 1)
            elif token == '[SEP]':
                input_ids.append(vocab_length + 2)
            else:
                input_ids.append(vocab_length + 3)

        yield (id, input_ids)


model = BiRNNWithPooling(num_inputs, num_time_steps, num_hidden, use_embedding_layer, embedding_matrix.shape)

init = tf.global_variables_initializer()

saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "birnn"))

data_gen = load_data()

print('Start...')

with tf.Session() as sess:
    sess.run(init)
    if os.path.isfile('./model/5/mlm-model-epoch5.ckpt.index'):
        print('Restoring weights...')
        saver.restore(sess, './model/5/mlm-model-epoch5.ckpt')
        print('Weights restored...')
    sess.run(model.trained_embedding.assign(model.saved_embeddings), {model.saved_embeddings: embedding_matrix})
    for id, data in chunks(data_gen):
        #print(id)
        data = np.asarray(data).reshape(-1, num_time_steps)
        input_ids = extract_data(data)

        feed_dict = {model.X: input_ids}

        out = sess.run(model.out, feed_dict)
        storage_repo.write(id, np.array2string(out[0], separator = ',').replace('\n', '').replace(' ', '').replace(',', ', '))

print('Done.')
