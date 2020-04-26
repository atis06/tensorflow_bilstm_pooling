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
import pickle
import simplejson as json

w2v_model = Word2Vec.load('/srv/project/encoder/model/w2v/enc-hu-oscar_sm-hun-spacy/enc-hu-oscar_sm-hun-spacy.w2v')
w2v_dim = 300

vocab_length = len(w2v_model.wv.vocab)

def get_embedding_matrix(w2v_model):
    embedding_matrix = np.zeros((len(w2v_model.wv.vocab) + 4, w2v_dim + 4))
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

def preprocess_data():

    input_ids_file = open("./data/input_ids", "r")
    #input_ids_file = open("./test", "r")
    for id, line in enumerate(input_ids_file):
        ids = line.split()
        embedded_vectors = np.asarray([embedding_matrix[int(id)] for id in ids]).tolist()
        yield embedded_vectors
    input_ids_file.close()

def dump_data():
    embedded_vectors_file = open("./data/embedded_vectors", "w")
    json.dump(preprocess_data(), embedded_vectors_file, iterable_as_array=True)
    embedded_vectors_file.close()

def get_data():

    embedded_vectors_file = open("./data/embedded_vectors", "r")
    json_f = json.load(embedded_vectors_file)
    for data in json_f:
        print(np.asarray(data).shape)


dump_data()

#get_data()
