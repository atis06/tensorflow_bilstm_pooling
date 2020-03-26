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
# Config
w2v_model = Word2Vec.load('./model/enc-hu-oscar-hun-spacy.w2v')
w2v_dim = 300

tokens_path = 'token'

batch_size = 3
min_sentence_length = 3
max_sentence_length = 15

# masking prediction for all data
masked_lm_prob = 0.8
# number of tokens to mask in a sequence
max_predictions_per_seq = 1


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
            chunk_input_ids.append(np.asarray([w2v_model.wv.vocab.get(token).index if w2v_model.wv.vocab.get(token) is not None else w2v_vocab_len for token in tokens]))

            ret_output_tokens, ret_masked_lm_positions, ret_masked_lm_labels = mask(tokens)
            chunk_masked_lm_positions.append(np.asarray(ret_masked_lm_positions))
            chunk_tokens.append(np.asarray(ret_output_tokens))
            chunk_masked_lm_weights.append([1.]*len(sentence) + [0.]*(max_sentence_length - len(sentence)))
            chunk_masked_lm_ids.append(np.asarray([w2v_model.wv.vocab.get(label).index if w2v_model.wv.vocab.get(label) is not None else w2v_vocab_len for label in ret_masked_lm_labels]))

        output_tokens.append(np.asarray(chunk_tokens))
        input_ids.append(np.asarray(chunk_input_ids))
        masked_lm_positions.append(np.asarray(chunk_masked_lm_positions))
        masked_lm_weights.append(np.asarray(chunk_masked_lm_weights))
        masked_lm_ids.append(np.asarray(chunk_masked_lm_ids))

    # yield np.asarray(output_tokens), np.asarray(input_ids), np.asarray(masked_lm_positions), np.asarray(masked_lm_weights), np.asarray(masked_lm_ids)
    return output_tokens, input_ids, masked_lm_positions, masked_lm_weights, masked_lm_ids


tokens, input_ids, masked_lm_positions, masked_lm_weights, masked_lm_ids = preprocess_data()

for data in tokens:
    print(data)
    print('###')

