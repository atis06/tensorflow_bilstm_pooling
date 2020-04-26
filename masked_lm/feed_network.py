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

#nltk.download('punkt')
from itertools import islice, chain, tee
import random
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
from model import BiRNNWithPooling
from tensorflow.python.client import device_lib
tf.compat.v1.disable_eager_execution()

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'


if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

# Config
save_model_to = './model/mlm_model.ckpt'

print('Load embedding model...')
w2v_model = Word2Vec.load('/srv/project/encoder/model/w2v/enc-hu-oscar_sm-hun-spacy/enc-hu-oscar_sm-hun-spacy.w2v')
w2v_dim = 300

print('Embedding model loaded.')

tokens_path = '../../repo/hungarian_spacy/'

batch_size = 32
min_sentence_length = 10

max_sentence_length = 100

# masking prediction for all data
masked_lm_prob = 0.15
# number of tokens to mask in a sequence
max_predictions_per_seq = math.ceil((max_sentence_length * masked_lm_prob) * 2) # from bert github

# network config
num_inputs = 1

num_hidden = 1024
learning_rate_start = 0.1
lr_decay = True
lr_decay_threshold = 0
dropout_keep_prob = 0.5
pooling = 'max'
use_embedding_layer = True

epochs = 5

random_seed = 733459
mask_padding = True

# End of config
random.seed(random_seed)

num_time_steps = 2 * max_sentence_length + 1

path = '.'
text_path = join(path, tokens_path)

all_file = [f for f in listdir(text_path) if isfile(join(text_path, f))]
all_file_len = len(all_file)
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

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_tokens():
    onlyfiles = all_file.copy()
    assert len(onlyfiles) == all_file_len
    for file in onlyfiles:
        in_file_name = join(text_path, file)
        in_file = open(in_file_name)
        try:
            try:
                whole_file_content = in_file.read().strip().replace('õ', 'ő').replace('û', 'ű').replace('ô', 'ő').replace('rdquo', '').replace('bdquo', '')
                doc_tokens = word_tokenize(whole_file_content)

                doc_tokens = [doc_tokens[i * max_sentence_length:(i + 1) * max_sentence_length] for i in range((len(doc_tokens) + max_sentence_length - 1) // max_sentence_length )]
                for tokens in doc_tokens:
                    if min_sentence_length <= len(tokens):
                        if len(tokens) < max_sentence_length:
                            yield tokens + ['[PAD]' for i in range(max_sentence_length - len(tokens))], file
                        else:
                            yield tokens, file
            except Exception as e:
                print(e)
        except Exception as e:
            print('error', e)
            in_file.close()


def chunks(iterable, rnd=False):
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
        if token == '[SEP]' or (not mask_padding and token == '[PAD]'):
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
                    masked_token = w2v_model.wv.index2word[random.randint(0, vocab_length - 1)]

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


def get_random_sentence(except_file):
    onlyfiles = all_file.copy()
    assert len(onlyfiles) == all_file_len
    onlyfiles.remove(except_file) # Can't open the data itself
    file = onlyfiles[random.randint(0, len(onlyfiles) - 1)]
    in_file_name = join(text_path, file)
    in_file = open(in_file_name)
    try:
        try:
            whole_file_content = in_file.read().strip().replace('õ', 'ő').replace('û', 'ű').replace('ô', 'ő').replace('rdquo', '').replace('bdquo', '')
            doc_tokens = word_tokenize(whole_file_content)

            while (len(doc_tokens) < max_sentence_length):
                onlyfiles.remove(file)
                file = onlyfiles[random.randint(0, len(onlyfiles) - 1)]
                in_file_name = join(text_path, file)
                in_file = open(in_file_name)
                whole_file_content = in_file.read().strip().replace('õ', 'ő').replace('û', 'ű').replace('ô', 'ő').replace('rdquo', '').replace('bdquo', '')
                doc_tokens = word_tokenize(whole_file_content)


            doc_tokens = [doc_tokens[i * max_sentence_length:(i + 1) * max_sentence_length] for i in range((len(doc_tokens) + max_sentence_length - 1) // max_sentence_length )]

            doc_tokens = [tokens for tokens in doc_tokens if min_sentence_length <= len(tokens)]

            returned_sentence = doc_tokens[random.randint(0, len(doc_tokens)-1)]
            return returned_sentence + ['[PAD]' for i in range(max_sentence_length - len(returned_sentence))]

        except Exception as e:
            print(e)
            sys.exit(1)
    except Exception as e:
        print('error', e)
        in_file.close()
        sys.exit(1)


def preprocess_data_gen():
    balancer = 0

    sentence_iterator = get_tokens()
    for A, B in pairwise(sentence_iterator):
        sentence_A, file_A = A
        sentence_B, file_B = B

        is_next_sentence = 1

        if file_A != file_B:
            sentence = sentence_A + ['[SEP]'] + sentence_B
            is_next_sentence = 0
            balancer += 1
        else:
            if balancer > 0:
                sentence = sentence_A + ['[SEP]'] + sentence_B
                is_next_sentence = 1
                balancer -= 1
            elif balancer == 0:
                if random.random() < 0.5:
                    sentence = sentence_A + ['[SEP]'] + get_random_sentence(file_B)
                    is_next_sentence = 0
                else:
                    sentence = sentence_A + ['[SEP]'] + sentence_B
                    is_next_sentence = 1
            else:
                print('ERROR: balancer is below 0!')
                sys.exit(1)


        ret_output_tokens, ret_masked_lm_positions, ret_masked_lm_labels = mask(sentence)
        input_ids = []
        for token in ret_output_tokens:
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

        masked_lm_weights_full_sentence = np.asarray([1. if token != "[PAD]" else 0. for token in sentence])
        masked_lm_weights_gathered = masked_lm_weights_full_sentence.take(ret_masked_lm_positions)
        masked_lm_ids = np.asarray(
            [w2v_model.wv.vocab.get(label).index if w2v_model.wv.vocab.get(label) is not None else vocab_length for
             label in ret_masked_lm_labels])

        yield np.asarray(ret_output_tokens), np.asarray(input_ids), np.asarray(ret_masked_lm_positions), np.asarray(
            masked_lm_weights_gathered), np.asarray(masked_lm_ids), is_next_sentence


def extract_data(data):
    output_tokens = []
    input_ids = []
    masked_lm_positions = []
    masked_lm_weights = []
    masked_lm_ids = []
    sentence_labels = []

    for elem in data:
        output_tokens.append(elem[0])
        input_ids.append(elem[1])
        masked_lm_positions.append(elem[2])
        masked_lm_weights.append(elem[3])
        masked_lm_ids.append(elem[4])
        sentence_labels.append(elem[5])

    return np.asarray(output_tokens), np.asarray(input_ids), np.asarray(masked_lm_positions), np.asarray(
        masked_lm_weights), np.asarray(masked_lm_ids), np.asarray(sentence_labels)


def preprocess_data():
    data_gen = preprocess_data_gen()

    tokens_file = open("./data/tokens", "ab")
    input_ids_file = open("./data/input_ids", "ab")
    masked_lm_positions_file = open("./data/masked_lm_positions", "ab")
    masked_lm_weights_file = open("./data/masked_lm_weights", "ab")
    masked_lm_ids_file = open("./data/masked_lm_ids", "ab")
    sentence_labels_file = open("./data/sentence_labels", "ab")

    for data in data_gen:
        data = np.asarray(data).reshape(-1, 6)
        tokens, input_ids, masked_lm_positions, masked_lm_weights, masked_lm_ids, sentence_labels = extract_data(data)

        np.savetxt(tokens_file, tokens, fmt='%s', encoding='utf8')
        #tokens_file.write(b"\n")

        np.savetxt(input_ids_file, input_ids, fmt='%u', encoding='utf8')
        #input_ids_file.write(b"\n")

        np.savetxt(masked_lm_positions_file, masked_lm_positions, fmt='%u', encoding='utf8')
        #masked_lm_positions_file.write(b"\n")

        np.savetxt(masked_lm_weights_file, masked_lm_weights, fmt='%u', encoding='utf8')
        #masked_lm_weights_file.write(b"\n")

        np.savetxt(masked_lm_ids_file, masked_lm_ids, fmt='%u', encoding='utf8')
        #masked_lm_ids_file.write(b"\n")

        np.savetxt(sentence_labels_file, sentence_labels, fmt='%u', encoding='utf8')
        #sentence_labels_file.write(b"\n")

    tokens_file.close()
    input_ids_file.close()
    masked_lm_positions_file.close()
    masked_lm_weights_file.close()
    masked_lm_ids_file.close()
    sentence_labels_file.close()


def load_data():
    tokens_file = open("./data/tokens", "r")
    input_ids_file = open("./data/input_ids", "r")
    masked_lm_positions_file = open("./data/masked_lm_positions", "r")
    masked_lm_weights_file = open("./data/masked_lm_weights", "r")
    masked_lm_ids_file = open("./data/masked_lm_ids", "r")
    sentence_labels_file = open("./data/sentence_labels", "r")

    for tokens, input_ids, masked_lm_positions, masked_lm_weights, masked_lm_ids, is_next_sentence in zip(tokens_file, input_ids_file, masked_lm_positions_file, masked_lm_weights_file, masked_lm_ids_file, sentence_labels_file):
        yield np.asarray(tokens.split()), np.asarray(input_ids.split()).astype(int), np.asarray(masked_lm_positions.split()).astype(int), np.asarray(
            masked_lm_weights.split()).astype(float), np.asarray(masked_lm_ids.split()).astype(int), int(is_next_sentence.rstrip('\n'))

if not os.listdir('./data'):
    print('Preprocess data...')
    preprocess_data()
    print('Preprocess done.')
else:
    print('Preprocessed data found.')

model = BiRNNWithPooling(num_inputs, num_time_steps, num_hidden, learning_rate_start, dropout_keep_prob, pooling,
                         use_embedding_layer, embedding_matrix.shape)

init = tf.global_variables_initializer()

saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "birnn"))

with tf.Session() as sess:
    prev_epoch_loss = 0
    sess.run(init)
    if os.path.isfile('./model/mlm-model.ckpt.index'):
        print('Restoring weights...')
        saver.restore(sess, 'model/mlm-model.ckpt')
        print('Weights restored...')
    sess.run(model.trained_embedding.assign(model.saved_embeddings), {model.saved_embeddings: embedding_matrix})
    for epoch in range(epochs):
        batches_num = 1
        epoch_loss = 0
        mlm_loss = 0
        next_sentence_loss = 0
        correct_mlm = 0
        all_mlm = 0
        correct_ns = 0
        all_ns = 0
        print('Epoch: ' + str(epoch + 1))
        random.seed(random_seed)
        data_gen = load_data()

        for i, data in enumerate(chunks(data_gen)):
            batches_num = i + 1
            '''if batches_num % 1000 == 0:
                print(batches_num)'''
            data = np.asarray(data).reshape(-1, 6)
            tokens, input_ids, masked_lm_positions, masked_lm_weights, masked_lm_ids, sentence_labels = extract_data(data)

            '''print(tokens)
            print(input_ids)
            print(masked_lm_positions)
            print(masked_lm_weights)
            print(masked_lm_ids)
            print(sentence_labels)
            print('###')'''
            feed_dict = {model.X: input_ids, model.positions: masked_lm_positions, model.label_ids: masked_lm_ids, model.label_weights: masked_lm_weights, model.sentence_labels: sentence_labels}
            model.train_masked_lm(sess, feed_dict)

            out = sess.run(model.out, feed_dict)

            # MLM accuracy
            pred_mlm = out[3]
            pred_mlm = np.argmax(pred_mlm, axis=-1).reshape(masked_lm_ids.shape)

            boolean_mask_mlm = pred_mlm == masked_lm_ids
            correct_predictions_mlm = np.sum(boolean_mask_mlm[masked_lm_weights == 1.])
            one_weighted_preds = np.count_nonzero(masked_lm_weights)

            correct_mlm += correct_predictions_mlm
            all_mlm += one_weighted_preds

            # Next sentence accuracy

            pred_ns = out[4]
            pred_ns = np.argmax(pred_ns, axis=1)

            boolean_mask_ns = pred_ns == sentence_labels
            correct_preds_ns = np.count_nonzero(boolean_mask_ns)

            correct_ns += correct_preds_ns
            all_ns += len(sentence_labels)

            #Losses

            per_batch_loss = out[0]
            epoch_loss += per_batch_loss

            per_batch_loss_mlm = out[1]
            mlm_loss += per_batch_loss_mlm

            per_batch_loss_ns = out[2]
            next_sentence_loss += per_batch_loss_ns

        epoch_loss = epoch_loss / batches_num
        mlm = mlm_loss / batches_num
        next_sentence = next_sentence_loss / batches_num
        mlm_acc = correct_mlm / all_mlm
        ns_acc = correct_ns / all_ns

        print('Epoch loss: ' + str(epoch_loss))
        print('MLM loss: ' + str(mlm))
        print('Next_sentence loss: ' + str(next_sentence))
        print('MLM accuracy: ' + str(mlm_acc))
        print('Next sentence accuracy: ' + str(ns_acc))
        print('Saving weights...')
        saver.save(sess, './model/' + str(epoch + 1) + '/mlm-model-epoch' + str(epoch + 1)  +  '.ckpt')
        print('Saved.')

        if lr_decay and epoch > 0 and (prev_epoch_loss - lr_decay_threshold) < epoch_loss:
            learning_rate_start = float(learning_rate_start / 5.)
            sess.run(model.learning_rate.assign(learning_rate_start))
            print('Learning rate divided by 5, new lr is: ' + str(learning_rate_start))
        prev_epoch_loss = epoch_loss

        print('------------------------------------------------------------')

    print('End.')
