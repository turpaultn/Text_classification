from StoreFiles import StoreFiles
import pandas as pd
import os
import shutil
from Constantes_public import model_files_folder, processed_files_folder, tensorboard_log_dir
from CreateTeamFile import CreateTeamFile
from Pipeline import TeamPipeline
import numpy as np
from Callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation, GRU, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, TensorBoard
import tensorflow as tf

from Log import logger


VERBOSE = 2
# Data
PERCENTAGE_UNREPRESENTED_LABEL = 0.
TEST_RATIO = 0.1
VALIDATION_RATIO = 0.2
LEMMATIZE = True
RANDOM_TEST_SAMPLES = False
FINETUNING_RATIO = 0.2
FINETUNING_EPOCHS = 5

# Word2vec
SIZE_W2VEC = 200
WINDOW = 10
MIN_COUNT = 100
WORKERS = 12
HIERARCHICAL_SOFTMAX = 0
SKIP_GRAM = 1
ITER = 15


class RunModel:
    def __init__(self, X, y, X_test, y_test):
        self.tp = TeamPipeline()
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test

        self.y, self.encoder = self.tp.encode_label(self.y, categorical=True)

        data_train, data_valid = self.tp.train_valid_split(self.X, self.y, test_size=VALIDATION_RATIO, shuffle=True)
        (self.X_train, self.y_train) = data_train
        (self.X_valid, self.y_valid) = data_valid

        _, (self.X_end, self.y_end) = self.tp.train_valid_split(self.X, self.y, test_size=FINETUNING_RATIO,
                                                                shuffle=False)

    def get_embedding_model(self):
        if W2V:
            embedding_model, model_filename = self.tp.get_word2vec_model(text_serie=self.X_train,
                                                                         size=SIZE_W2VEC, window=WINDOW,
                                                                         min_count=MIN_COUNT,
                                                                         workers=WORKERS, hs=HIERARCHICAL_SOFTMAX,
                                                                         sg=SKIP_GRAM,
                                                                         iter=ITER, alpha=0.05)

        else:
            embedding_model, model_filename = self.tp.get_tokenizer(self.X_train)
        return embedding_model, model_filename

    def evaluate_model_train_valid(self, data, model, embedding_model):
        assert len(data) == 2, "data must be a tuple (X, y)"
        X, y = data
        model.evaluate_generator(self.tp.data_to_batch_generator(X, y, shuffle=False,
                                                                 embedding_model=embedding_model,
                                                                 batch_size=BATCH_SIZE,
                                                                 len_padding=LEN_PADDING),
                                 steps=len(X) / BATCH_SIZE,
                                 workers=4)

    def predict_model(self, X, model, embedding_model):
        predicted = model.predict_generator(self.tp.data_to_batch_generator(X, shuffle=False,
                                                                            embedding_model=embedding_model,
                                                                            batch_size=BATCH_SIZE,
                                                                            len_padding=LEN_PADDING),
                                            steps=len(X) / BATCH_SIZE,
                                            workers=4)
        return predicted

    def finetune(self, data, model_rnn, embedding_model, epochs=1):
        assert len(data) == 2, "data must be (X,y)"
        X, y = data
        model_rnn.fit_generator(self.tp.data_to_batch_generator(X, y, shuffle=True,
                                                                embedding_model=embedding_model,
                                                                batch_size=BATCH_SIZE,
                                                                len_padding=LEN_PADDING),

                                steps_per_epoch=len(X) / BATCH_SIZE,
                                epochs=epochs, verbose=VERBOSE, workers=12,
                                )

        return model_rnn

    def print_accuracy(self, y_test, predicted, encoder):
        nb_classes = len(set(predicted.argmax(axis=1)))
        if nb_classes < 3:
            best_predictions = encoder.inverse_transform(predicted.argmax(axis=1))
        else:
            best_predictions = encoder.inverse_transform(np.argpartition(predicted, -3)[:, -3:])
        count = 0
        for i, value in enumerate(y_test):
            if value in best_predictions[i]:
                count += 1
        print("test accuracy with 3 best: {}".format(count / len(y_test)))
        print("test accuracy: {}".format(
            sum(encoder.inverse_transform(predicted.argmax(axis=1)) == y_test) / len(predicted)))

    def remove_models(self, model_checkpoint):
        model_checkpoint.remove()
        shutil.rmtree(os.path.join(model_files_folder, tensorboard_log_dir, model_checkpoint.filename))

    def model(self, batch_size, layers, len_padding, loss, optimizer, layers_type, dropout, W2V, bidirectional):
        embedding_model, model_filename = self.get_embedding_model()

        params_model = {
            "loss": loss,
            "optimizer": optimizer,
            "bs": batch_size,
            "layers": layers,
            "len_padding": len_padding,
            "embedding_model": model_filename,
            "layers_type": layers_type,
            "dropout": dropout,
            "bidirectional": bidirectional
        }

        logger.info(params_model)
        sf = StoreFiles('model')
        fname = sf.get_fname_by_part_of_params(params_model)
        if fname is not None:
            model = load_model(os.path.join(model_files_folder, fname))
            logger.info('predict ...')
            print("training accuracy: {}".format(
                  self.evaluate_model_train_valid((self.X_train, self.y_train), model, embedding_model)))
            print("validation accuracy: {}".format(
                  self.evaluate_model_train_valid((self.X_valid, self.y_valid), model, embedding_model)))

            predicted = self.predict_model(self.X_test, model, embedding_model)
            self.print_accuracy(self.y_test, predicted, self.encoder)

            model = self.finetune((self.X_end, self.y_end), model, embedding_model, epochs=5)

            predicted = self.predict_model(self.X_test, model, embedding_model)
            self.print_accuracy(self.y_test, predicted, self.encoder)
            return

        seq_shape = (batch_size, len_padding, SIZE_W2VEC)

        if layers_type == 'LSTM':
            recurrent_layer = LSTM
        elif layers_type == 'GRU':
            recurrent_layer = GRU
        else:
            raise AttributeError('Wrong attribute for layer: {}'.format(layers_type))

        model = Sequential()
        if not W2V:
            model.add(Embedding(len(embedding_model.word_index), SIZE_W2VEC,
                                batch_input_shape=(seq_shape[0], seq_shape[1])))

        model.add(Dropout(dropout, noise_shape=(seq_shape[0], 1, seq_shape[2]),
                          batch_input_shape=seq_shape))

        def add_recurrent_layer(model, bidirectional, return_sequences):
            if bidirectional:
                model.add(Bidirectional(recurrent_layer(layer_size, batch_input_shape=seq_shape,
                                          return_sequences=return_sequences)))
            else:
                model.add(recurrent_layer(layer_size, batch_input_shape=seq_shape,
                                          return_sequences=return_sequences))

        for layer_size in layers[:-1]:
            add_recurrent_layer(model, bidirectional, return_sequences=True)

        layer_size = layers[-1]
        add_recurrent_layer(model, bidirectional, return_sequences=False)

        model.add(Dense(self.y.shape[1]))
        # we take the full number of teams here, even if some of them does not appear in the training set,
        # it is easier for further prediction
        # TODO put only train set here, and adapt test evaluation (accuracy, etc...)
        model.add(Activation('softmax'))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        model.summary()

        try:
            model_checkpoint = ModelCheckpoint(monitor='val_loss', verbose=VERBOSE, mode='auto', period=1,
                                               params=params_model)
            callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=4),
                         model_checkpoint,
                         TensorBoard(
                            log_dir=os.path.join(model_files_folder, tensorboard_log_dir, model_checkpoint.filename),
                            batch_size=batch_size)
                         ]
            logger.info('training ...')
            model.fit_generator(self.tp.data_to_batch_generator(self.X_train, self.y_train, shuffle=True,
                                                                embedding_model=embedding_model,
                                                                batch_size=batch_size,
                                                                len_padding=len_padding),
                                steps_per_epoch=len(self.X_train) / batch_size,
                                validation_data=self.tp.data_to_batch_generator(self.X_valid, self.y_valid,
                                                                                shuffle=True,
                                                                                embedding_model=embedding_model,
                                                                                batch_size=batch_size,
                                                                                len_padding=len_padding),
                                validation_steps=len(self.X_valid) / batch_size,
                                epochs=100, verbose=VERBOSE, workers=12,
                                callbacks=callbacks)

            logger.info('predict ...')
            predicted = self.predict_model(self.X_test, model, embedding_model)
            self.print_accuracy(self.y_test, predicted, self.encoder)

        except Exception as e:
            logger.error(e)
            self.remove_models(model_checkpoint)
            logger.warn('removed file: {} because train did not go to the end'.format(model_checkpoint.filename))

        model = self.finetune((self.X_end, self.y_end), model, embedding_model, FINETUNING_EPOCHS)
        logger.info('predict ...')
        predicted = self.predict_model(self.X_test, model, embedding_model)
        self.print_accuracy(self.y_test, predicted, self.encoder)


if __name__ == '__main__':
    logger.debug('training model ...')

    # Do it for your data
    def get_data():
        sf = StoreFiles('preprocessing')
        # TODO change StoreFiles and/or CreateTeamFile to be able to get both (train and test)
        # name together and store them together
        # Here assumption that train and test are both present if train present, but can lead to mistakes
        fname_train = sf.get_fname(lemmatize=LEMMATIZE, test_ratio=TEST_RATIO, dataset_type='train',
                                   random_test_samples=RANDOM_TEST_SAMPLES)
        fname_test = sf.get_fname(lemmatize=LEMMATIZE, test_ratio=TEST_RATIO, dataset_type='test',
                                  random_test_samples=RANDOM_TEST_SAMPLES)
        if fname_train is not None:
            logger.info('file read {}'.format(fname_train))
            data_train = pd.read_csv(os.path.join(processed_files_folder, fname_train), index_col=0, encoding='latin1')
            data_test = pd.read_csv(os.path.join(processed_files_folder, fname_test), index_col=0, encoding='latin1')
        else:
            data_train, data_test = CreateTeamFile(lemmatize=LEMMATIZE, test_ratio=TEST_RATIO,
                                                   random_test_samples=RANDOM_TEST_SAMPLES).create()

        if PERCENTAGE_UNREPRESENTED_LABEL > 0:
            data_train = TeamPipeline.drop_unrepresented_labels(data_train, 'C_EQUIPE',
                                                                percentage_drop=PERCENTAGE_UNREPRESENTED_LABEL)

        logger.debug('take only 10 values to test the program')
        data_test = data_test.iloc[:10]
        data_train = data_train.iloc[:10]

        X = data_train['DE_SYMPAPPEL'].fillna('')
        y = data_train['C_EQUIPE']
        X_test = data_test['DE_SYMPAPPEL'].fillna('')
        y_test = data_test['C_EQUIPE']

        return (X, y), (X_test, y_test)


    (X, y), (X_test, y_test) = get_data()

    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # also tried [8, 16, 256, 128, 32]
    bs_list = [2]

    # also tried [[80], [256], [128, 128]]
    layers_list = [[2], [128]]

    # also tried [50, 100, 150]
    len_pad_list = [2]

    # also tried ['rmsprop', 'adagrad']
    optimizer_list = ['adam']
    loss_list = ['categorical_crossentropy']

    # also tried ['GRU']
    layers_type_list = ['LSTM']
    true_false = [True, False]
    dropout_list = [0.4, 0]

    for BATCH_SIZE in bs_list:
        for LAYERS in layers_list:
            for LEN_PADDING in len_pad_list:
                for LOSS in loss_list:
                    for OPTIMIZER in optimizer_list:
                        for LAYERS_TYPE in layers_type_list:
                            for DROPOUT in dropout_list:
                                for W2V in true_false:
                                    for BIDIRECTIONAL in true_false:
                                        RunModel(X, y, X_test, y_test).model(BATCH_SIZE, LAYERS, LEN_PADDING,
                                                                             LOSS, OPTIMIZER,LAYERS_TYPE, DROPOUT,
                                                                             W2V, BIDIRECTIONAL)
