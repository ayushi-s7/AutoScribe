import re
import numpy as np
import keras
import tensorflow as tf
import pickle

import re

class Text:
    def __init__(self, input_text, token2ind=None, ind2token=None):
        self.content = input_text
        self.tokens, self.tokens_distinct = self.tokenize()

        if token2ind != None and ind2token != None:
            self.token2ind, self.ind2token = token2ind, ind2token
        else:
            self.token2ind, self.ind2token = self.create_word_mapping(self.tokens_distinct)

        self.tokens_ind = [self.token2ind[token] if token in self.token2ind.keys() else self.token2ind['<| unknown |>']
                           for token in self.tokens]

    def __repr__(self):
        return self.content

    def __len__(self):
        return len(self.tokens_distinct)

    @staticmethod
    def create_word_mapping(values_list):
        values_list.append('<| unknown |>')
        value2ind = {value: ind for ind, value in enumerate(values_list)}
        ind2value = dict(enumerate(values_list))
        return value2ind, ind2value

    def preprocess(self):
        punctuation_pad = '!?.,:-;'
        punctuation_remove = '"()_\n'

        self.content_preprocess = re.sub(r'(\S)(\n)(\S)', r'\1 \2 \3', self.content)
        self.content_preprocess = self.content_preprocess.translate(str.maketrans('', '', punctuation_remove))
        self.content_preprocess = self.content_preprocess.translate(
            str.maketrans({key: ' {0} '.format(key) for key in punctuation_pad}))
        self.content_preprocess = re.sub(' +', ' ', self.content_preprocess)
        self.content = self.content_preprocess.strip()

    def tokenize(self):
        self.preprocess()
        tokens = self.content.split(' ')
        return tokens, list(set(tokens))

class Sequences():
    def __init__(self, text_object, max_len, step):
        self.tokens_ind = text_object.tokens_ind
        self.max_len = max_len
        self.step = step
        self.sequences, self.next_words = self.create_sequences()

    def __repr__(self):
        return 'Sequence object of max_len: %d and step: %d' % (self.max_len, self.step)

    def __len__(self):
        return len(self.sequences)

    def create_sequences(self):
        sequences = []
        next_words = []

        for i in range(0, len(self.tokens_ind) - self.max_len, self.step):
            sequences.append(self.tokens_ind[i: i +self.max_len])
            next_words.append(self.tokens_ind[ i +self.max_len])
        return sequences, next_words

class ModelPredict():
    def __init__(self, model, prefix, token2ind, ind2token, max_len, embedding=False):
        self.model = model
        self.token2ind, self.ind2token = token2ind, ind2token
        self.max_len = max_len
        self.prefix = prefix
        self.tokens_ind = prefix.tokens_ind.copy()
        self.embedding = embedding

    def __repr__(self):
        return self.prefix.content

    def single_data_generation(self):
        single_sequence = np.zeros((1, self.max_len, len(self.token2ind)), dtype=bool)
        prefix = self.tokens_ind[-self.max_len:]

        for i, s in enumerate(prefix):
            single_sequence[0, i, s] = 1
        return single_sequence

    def model_predict(self):
        if self.embedding:
          model_input = np.array(self.tokens_ind).reshape(1,-1)
        else:
          model_input = self.single_data_generation()
        return self.model.predict(model_input)[0]

    @staticmethod
    def add_prob_temperature(prob, temperature=1):
        prob = prob.astype(float)
        prob_with_temperature = np.exp(np.where(prob == 0, 0, np.log(prob + 1e-10)) / temperature)
        prob_with_temperature /= np.sum(prob_with_temperature)
        return prob_with_temperature

    @staticmethod
    def reverse_preprocess(text):
        text_reverse = re.sub(r'\s+([!?"\'().,;-])', r'\1', text)
        text_reverse = re.sub(' +', ' ', text_reverse)
        return text_reverse

    def return_next_word(self, temperature=1, as_word=False, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        prob = self.model_predict()
        prob_with_temperature=self.add_prob_temperature(prob,temperature)
        next_word = np.random.choice(len(prob_with_temperature), p=prob_with_temperature)
        if as_word:
            return self.ind2token[next_word]
        else:
            return next_word
        
class TextDataGenerator(keras.utils.Sequence):
    def __init__(self, sequences, next_words, sequence_length, vocab_size, batch_size=32, shuffle=True, embedding=False):
        self.batch_size = batch_size
        self.sequences = sequences
        self.next_words = next_words
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.shuffle = shuffle
        self.embedding = embedding
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.sequences) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        sequences_batch = [self.sequences[k] for k in indexes]
        next_words_batch = [self.next_words[k] for k in indexes]

        if self.embedding:
          X = np.array(sequences_batch)
          y = keras.utils.to_categorical(next_words_batch, num_classes=self.vocab_size)
        else:
          X, y = self.__data_generation(sequences_batch, next_words_batch)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.sequences))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, sequences_batch, next_words_batch):
        X = np.zeros((self.batch_size, self.sequence_length, self.vocab_size), dtype=bool)
        y = np.zeros((self.batch_size, self.vocab_size), dtype=bool)

        for i, seq in enumerate(sequences_batch):
            for j, word in enumerate(seq):
                X[i, j, word] = 1
                y[i, next_words_batch[i]] = 1
        return X, y
		
# loading
with open('token2ind.pickle', 'rb') as handle:
    token2ind = pickle.load(handle)

# # loading
with open('ind2token.pickle', 'rb') as handle:
    ind2token= pickle.load(handle)


def autocomplete(input_prefix):
    text_prefix = Text(input_prefix, token2ind, ind2token)

    loaded_model = tf.keras.models.load_model('model1.h5')

    pred = ModelPredict(loaded_model, text_prefix, token2ind, ind2token, max_len=4)
    print(pred.return_next_word(temperature=1,as_word=True,random_seed=42))
    return pred.return_next_word(temperature=1,as_word=True,random_seed=42)