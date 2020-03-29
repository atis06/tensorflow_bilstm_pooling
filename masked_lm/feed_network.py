from os import listdir
from os.path import isfile, join
from gensim.models import Word2Vec
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import collections

nltk.download('punkt')
from itertools import islice, chain
import random
import tensorflow as tf
from masked_lm.model import BiRNNWithPooling

# Config
save_model_to = './model/mlm_model.ckpt'

w2v_model = Word2Vec.load('./model/enc-hu-oscar-hun-spacy.w2v')
w2v_dim = 300

tokens_path = 'token'

batch_size = 5
min_sentence_length = 5

max_sentence_length = 14

# masking prediction for all data
masked_lm_prob = 0.8
# number of tokens to mask in a sequence
max_predictions_per_seq = 2

# network config
num_inputs = 1
num_time_steps = max_sentence_length
num_hidden = 512
learning_rate = 0.001
dropout_keep_prob = 1
pooling = 'max'
use_embedding_layer = True

epochs = 1


def get_embedding_matrix(w2v_model):
    embedding_matrix = np.zeros((len(w2v_model.wv.vocab), w2v_dim))
    for i in range(len(w2v_model.wv.vocab)):
        embedding_vector = w2v_model.wv[w2v_model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


embedding_matrix = get_embedding_matrix(w2v_model)


def get_tokens():
    path = '.'
    text_path = join(path, tokens_path)

    onlyfiles = [f for f in listdir(text_path) if isfile(join(text_path, f))]
    for file in onlyfiles:
        in_file_name = join(text_path, file)
        in_file = open(in_file_name)
        try:

            try:
                for line in in_file:
                    clean_line = line.strip()
                    if clean_line != "":
                        tokens = word_tokenize(clean_line)
                        if min_sentence_length <= len(tokens):
                            yield tokens
            except Exception as e:
                print(e)
        except Exception as e:
            print('error', e)
            in_file.close()


def chunks(iterable, rnd=True):
    iterator = iter(iterable)
    for first in iterator:
        if rnd:
            std = sorted(list(chain([first], islice(iterator, batch_size - 1))), key=lambda k: random.random())
            yield std
        else:
            yield list(chain([first], islice(iterator, batch_size - 1)))


def mask(tokens):
    MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                              ["index", "label"])

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        # Don't mask padding
        if token == '[PAD]':
            continue

        cand_indexes.append([i])

    random.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = w2v_model.wv.index2word[random.randint(0, len(w2v_model.wv.vocab) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return output_tokens, masked_lm_positions, masked_lm_labels


def preprocess_data_gen():
    for sentence in get_tokens():
        if len(sentence) < max_sentence_length:
            tokens = sentence + ['[PAD]' for i in range(max_sentence_length - len(sentence))]
        else:
            tokens = sentence[0:max_sentence_length]

        w2v_vocab_len = len(w2v_model.wv.vocab)
        input_ids = [w2v_model.wv.vocab.get(token).index if w2v_model.wv.vocab.get(token) is not None else w2v_vocab_len
                     for token in tokens]

        ret_output_tokens, ret_masked_lm_positions, ret_masked_lm_labels = mask(tokens)
        masked_lm_weights_full_sentence = np.asarray(
            [1.] * len(sentence) + [0.] * (max_sentence_length - len(sentence)))
        masked_lm_weights_gathered = masked_lm_weights_full_sentence.take(ret_masked_lm_positions)
        masked_lm_ids = np.asarray(
            [w2v_model.wv.vocab.get(label).index if w2v_model.wv.vocab.get(label) is not None else w2v_vocab_len for
             label in ret_masked_lm_labels])

        yield np.asarray(ret_output_tokens), np.asarray(input_ids), np.asarray(ret_masked_lm_positions), np.asarray(
            masked_lm_weights_gathered), np.asarray(masked_lm_ids)


def preprocess_data():
    output_tokens = []
    input_ids = []
    masked_lm_positions = []
    masked_lm_weights = []
    masked_lm_ids = []

    for chunk in chunks(get_tokens()):
        chunk_tokens = []
        chunk_input_ids = []
        chunk_masked_lm_positions = []
        chunk_masked_lm_weights = []
        chunk_masked_lm_ids = []
        for sentence in chunk:
            if len(sentence) < max_sentence_length:
                tokens = sentence + ['[PAD]' for i in range(max_sentence_length - len(sentence))]
            else:
                tokens = sentence[0:max_sentence_length]

            w2v_vocab_len = len(w2v_model.wv.vocab)
            chunk_input_ids.append(
                [w2v_model.wv.vocab.get(token).index if w2v_model.wv.vocab.get(token) is not None else w2v_vocab_len for
                 token in tokens])

            ret_output_tokens, ret_masked_lm_positions, ret_masked_lm_labels = mask(tokens)
            chunk_masked_lm_positions.append(ret_masked_lm_positions)
            chunk_tokens.append(ret_output_tokens)
            masked_lm_weights_full_sentence = np.asarray(
                [1.] * len(sentence) + [0.] * (max_sentence_length - len(sentence)))
            masked_lm_weights_gathered = masked_lm_weights_full_sentence.take(ret_masked_lm_positions)
            chunk_masked_lm_weights.append(masked_lm_weights_gathered)
            chunk_masked_lm_ids.append(np.asarray(
                [w2v_model.wv.vocab.get(label).index if w2v_model.wv.vocab.get(label) is not None else w2v_vocab_len for
                 label in ret_masked_lm_labels]))

        output_tokens.append(np.asarray(chunk_tokens))
        input_ids.append(np.asarray(chunk_input_ids))
        masked_lm_positions.append(np.asarray(chunk_masked_lm_positions))
        masked_lm_weights.append(np.asarray(chunk_masked_lm_weights))
        masked_lm_ids.append(np.asarray(chunk_masked_lm_ids))

        yield np.asarray(output_tokens), np.asarray(input_ids), np.asarray(masked_lm_positions), np.asarray(
            masked_lm_weights), np.asarray(masked_lm_ids)
        # return output_tokens, np.asarray(input_ids), masked_lm_positions, masked_lm_weights, masked_lm_ids


def extract_data(data):
    output_tokens = []
    input_ids = []
    masked_lm_positions = []
    masked_lm_weights = []
    masked_lm_ids = []

    for elem in data:
        output_tokens.append(elem[0])
        input_ids.append(elem[1])
        masked_lm_positions.append(elem[2])
        masked_lm_weights.append(elem[3])
        masked_lm_ids.append(elem[4])

    return np.asarray(output_tokens), np.asarray(input_ids), np.asarray(masked_lm_positions), np.asarray(
        masked_lm_weights), np.asarray(masked_lm_ids)


model = BiRNNWithPooling(num_inputs, num_time_steps, num_hidden, learning_rate, dropout_keep_prob, pooling,
                         use_embedding_layer, embedding_matrix)

init = tf.global_variables_initializer()

saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "birnn"))

with tf.Session() as sess:
    sess.run(init)
    sess.run(model.trained_embedding.assign(model.saved_embeddings), {model.saved_embeddings: embedding_matrix})
    for epoch in range(epochs):
        c = 0
        epoch_loss = 0
        print('Epoch: ' + str(epoch + 1))
        data_gen = preprocess_data_gen()

        for i, data in enumerate(chunks(data_gen)):
            c = i
            data = np.asarray(data).reshape(-1, 5)
            tokens, input_ids, masked_lm_positions, masked_lm_weights, masked_lm_ids = extract_data(data)

            feed_dict = {model.X: input_ids, model.positions: masked_lm_positions, model.label_ids: masked_lm_ids,
                         model.label_weights: masked_lm_weights}
            model.train_masked_lm(sess, feed_dict)

            pred = sess.run(model.out, feed_dict)[1]
            pred = np.argmax(pred, axis=-1)

            per_batch_loss = sess.run(model.out, feed_dict)[0]
            epoch_loss += per_batch_loss

        epoch_loss = epoch_loss / c
        print('Epoch loss: ' + str(epoch_loss))
        print('------------------------------------------------------------')

    print('Saving weights...')
    saver.save(sess, './model/mlm-model.ckpt')
    print('Saved.')
    print('End.')
