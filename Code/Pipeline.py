from gensim.models import word2vec
import numpy as np
import re
import os
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import sklearn.utils
from sklearn.model_selection import train_test_split
from ThreadsafeIter import threadsafe_generator
from StoreFiles import StoreFiles
import pandas as pd
import gensim
import keras
from keras.preprocessing.text import Tokenizer
import pickle
from Constantes_public import model_files_folder

from Log import logger


class TeamPipeline:
    def __init__(self):
        pass

    @staticmethod
    def drop_unrepresented_labels(df, label_col, percentage_drop=0.1):
        categories_counts = pd.value_counts(df[label_col])
        reduced_categories_counts = categories_counts[(categories_counts / categories_counts.sum())*100.
                                                      > percentage_drop]
        percentage_keep = reduced_categories_counts.sum() / categories_counts.sum() * 100
        logger.info("We keep {:.2f} % of the data and {} labels".format(percentage_keep,
                                                                        len(reduced_categories_counts)))
        result = df[df[label_col].isin(reduced_categories_counts.index)]
        result = result.reset_index(drop=True)
        return result

    @staticmethod
    def split_and_lower(text):
        cleaned_sentence = []
        words = re.split("\W", text)
        for word in words:
            if word != '':
                if not word.isupper():
                    word = word.lower()
                cleaned_sentence.append(word)
        return cleaned_sentence

    def get_word2vec_model(self, text_serie, **kwargs):
        sf = StoreFiles('model')
        fname = sf.get_fname(type="word2vec", **kwargs)
        if fname is not None:
            model = word2vec.Word2Vec.load(os.path.join(model_files_folder, fname))

        else:
            logger.debug('training word2vec ...')
            sentences = []
            sentences.extend(text_serie.fillna('').apply(self.split_and_lower))
            import time
            t1 = time.time()
            model = word2vec.Word2Vec(sentences, **kwargs)
            # size=100, window=5, min_count=100, workers=8, hs=1, sg=1, iter=5)
            logger.info('time running word2vec model : {}'.format(time.time() - t1))
            # if save_model:
            fname = sf.fname_and_store(type="word2vec", **kwargs)
            model.save(os.path.join(model_files_folder, fname))
        return model, fname

    def get_tokenizer(self, text_serie, **kwargs):
        sf = StoreFiles('model')
        fname = sf.get_fname(type="tokenizer", **kwargs)
        if fname is not None:
            try:
                tokenizer = pickle.load(open(fname, 'rb'))
            except FileNotFoundError as e:
                logger.error("trying to reload the file")
                tokenizer = pickle.load(open(fname, 'rb'))

        else:
            logger.info('Tokenizer ... ')
            model_filename = sf.fname_and_store(type="tokenizer", **kwargs)
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(text_serie)
            pickle.dump(tokenizer, open(os.path.join(model_files_folder, model_filename), 'wb'))
            logger.info('Tokenized')

        return tokenizer, fname

    def sentence_to_word2vec_embedding(self, word2vec_model, sentence):
        result = []
        if type(sentence) is float:
            return result
        sent_split = sentence.split()
        for word in sent_split:
            try:
                result.append(word2vec_model.wv.word_vec(word))
            except KeyError:
                continue
        # if len(result) == 0:
        #     result.append(np.zeros(word2vec_model.layer1_size))
        return result

    def encode_label(self, labels, categorical=True):
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        if categorical:
            labels = to_categorical(labels)
        return labels, label_encoder

    def train_valid_split(self, X, y, test_size=0.2, shuffle=True):
        assert len(X) == len(y), 'X and y in train valid split must have same length'
        if shuffle:
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size)
        else:
            if test_size > 1:
                idx = test_size
            else:
                idx = int(len(X) * (1 - test_size))
            X_train, X_valid = X[:idx], X[idx:]
            y_train, y_valid = y[:idx], y[idx:]
        logger.info("length train: {}  and length valid: {}".format(len(X_train), len(X_valid)))
        return (X_train, y_train), (X_valid, y_valid)

    @threadsafe_generator
    def data_to_batch_generator(self, X, y=None, shuffle=False, embedding_model=None, batch_size=32, len_padding=100):
        while True:
            if shuffle:
                if y is not None:
                    X, y = sklearn.utils.shuffle(X, y)
                else:
                    raise Exception("if you shuffle X without y you will not be able to get the order for y")

            for i in range(0, len(X), batch_size):
                batch = X[i:(i + batch_size)]

                if batch_size > len(X):
                    raise RuntimeError("Impossible to compute a batch size greater than the size of data")
                # TODO: find a way to include last the last batch which has a different size
                # if (i + batch_size) > len(X):
                #     logger.warn("Taking random rows to complete last batch")
                #     idx = np.random.randint(len(X), size=(i+batch_size) - len(X))
                #     batch = pd.concat([X[i:(i + batch_size)], X[idx]], axis=0)

                if type(embedding_model) is gensim.models.word2vec.Word2Vec:
                    batch = np.array(batch.apply(lambda sent: self.sentence_to_word2vec_embedding(embedding_model, sent)))
                elif type(embedding_model) is keras.preprocessing.text.Tokenizer:
                    if batch_size == 1:
                        raise NotImplementedError("If you do not use word2vec, take a batch size greater than 1")
                    batch = embedding_model.texts_to_sequences(batch)
                else:
                    # Todo find a way to add new embedding models
                    raise NotImplementedError('Need to implement behavior when no embedding or other embedings')
                    pass
                if len_padding is not None:
                    batch = pad_sequences(batch, maxlen=len_padding, dtype=np.float32,
                                          padding='pre', truncating='post')

                if y is not None:
                    batch_labels = y[i:(i + batch_size)]
                    yield batch, batch_labels
                else:
                    yield batch


if __name__ == '__main__':
    for W2V in [True, False]:
        df = pd.DataFrame([['bonjour je avoir un problème ici', "team1"],
                           ['nous aller le réosudre de ici là', "team2"],
                           ["nous vouloir améliorer", "team1"],
                           ["je être maintenant devant un problème", "team3"]], columns=["Text", "Team"])
        print(df)
        X = df.Text
        y = df.Team
        tp = TeamPipeline()
        y, encoder = tp.encode_label(y, categorical=True)
        (X_train, y_train), (X_valid, y_valid) = tp.train_valid_split(X, y)

        if W2V:
            embedding_model, fname_embedding = tp.get_word2vec_model(X_train, size=3, window=2,
                                                    min_count=0)
        else:
            embedding_model, fname_embedding = tp.get_tokenizer(X_train)

        gen = tp.data_to_batch_generator(X_train, y_train, shuffle=True, embedding_model=embedding_model,
                                         batch_size=32, len_padding=10)

        for i in gen:
            print(i)
            print(i[0].shape)
            break
