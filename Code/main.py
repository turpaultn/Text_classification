from StoreFiles import StoreFiles
import pandas as pd
import os
import shutil
from Constantes_public import model_files_folder, processed_files_folder, tensorboard_log_dir
from Pipeline import TeamPipeline
import numpy as np
from Callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation, GRU, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, TensorBoard
import tensorflow as tf

from Log import logger

# printing
VERBOSE = 0
# Data
PERCENTAGE_UNREPRESENTED_LABEL = 0.
TEST_RATIO = 0.2
VALIDATION_RATIO = 0.2
LEMMATIZE = True
RANDOM_TEST_SAMPLES = False
FINETUNING_RATIO = 0.2
FINETUNING_EPOCHS = 5

# Word2vec
SIZE_W2VEC = 200
WINDOW = 10
# For small example purpose only, it was 100 during my experiments
MIN_COUNT = 1
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

    def predict_class(self, X, model, embedding_model):
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
        logger.debug(len(encoder.inverse_transform(predicted.argmax(axis=1))))
        logger.debug(len(y_test))
        print("test accuracy: {}".format(
            sum(encoder.inverse_transform(predicted.argmax(axis=1)) == y_test) / len(predicted)))

    def apply_prediction(self, batch_size, model, embedding_model):
        regularize_test_len = len(self.X_test) % batch_size
        X_test = self.X_test
        y_test = self.y_test
        if not regularize_test_len == 0:
            X_test = self.X_test.iloc[:-regularize_test_len]
            y_test = self.y_test.iloc[:-regularize_test_len]
        predicted = self.predict_class(X_test, model, embedding_model)
        self.print_accuracy(y_test, predicted, self.encoder)

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
        fname = sf.get_fname(type=None, **params_model)
        if fname is not None:
            model = load_model(os.path.join(model_files_folder, fname))
            logger.info('predict ...')
            print("training accuracy: {}".format(
                  self.evaluate_model_train_valid((self.X_train, self.y_train), model, embedding_model)))
            print("validation accuracy: {}".format(
                  self.evaluate_model_train_valid((self.X_valid, self.y_valid), model, embedding_model)))

            self.apply_prediction(batch_size, model, embedding_model)

            model = self.finetune((self.X_end, self.y_end), model, embedding_model, epochs=5)

            self.apply_prediction(batch_size, model, embedding_model)
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


        ### (batch_size, len_padding, SIZE_W2VEC)
        model.add(Dropout(dropout, noise_shape=(seq_shape[0], 1, seq_shape[2]),
                          batch_input_shape=seq_shape))
        ### (batch_size, len_padding, SIZE_W2VEC)
        def add_recurrent_layer(model, bidirectional, return_sequences):
            if bidirectional:
                model.add(Bidirectional(recurrent_layer(layer_size, batch_input_shape=seq_shape,
                                          return_sequences=return_sequences)))
            else:
                model.add(recurrent_layer(layer_size, batch_input_shape=seq_shape,
                                          return_sequences=return_sequences))
        ### (batch_size, len_padding, SIZE_W2VEC)

        for layer_size in layers[:-1]:
            add_recurrent_layer(model, bidirectional, return_sequences=True)

        ### (batch_size, SIZE_W2VEC)
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


        except Exception as e:
            logger.error(e)
            self.remove_models(model_checkpoint)
            logger.warn('removed file: {} because train did not go to the end'.format(model_checkpoint.filename))

        model = self.finetune((self.X_end, self.y_end), model, embedding_model, FINETUNING_EPOCHS)
        logger.info('predict ...')
        self.apply_prediction(batch_size, model, embedding_model)


if __name__ == '__main__':
    logger.debug('training model ...')

    #useless, allow to create a fake csv file, but you should remove this and load your own csv file
    fake_data_path = os.path.join(processed_files_folder, "data.csv")

    def create_fake_data():
        logger.debug("creating file")
        df = pd.DataFrame([['bonjour je avoir un problème ici', "team1"],
                           ['nous aller le réosudre de ici là', "team2"],
                           ["nous vouloir améliorer", "team1"],
                           ["je être maintenant devant un problème", "team3"],
                           ["je suis un faux texte", "team2"],
                           ["ce texte est en français car le code est adapté pour du français", "team1"],
                           ["Voici un texte ave cune fautes de frappe et d'orthographe", "team3"],
                           ["Les textes de cet exemples ne donneront aucun résultat", "team3"],
                           ["Mais pas de problème, on va y arriver tout de même", "team1"],
                           ["Encore un texte inutile, remplacez par les votres", "team2"],
                           ["taille inutile", "team1"],
                           ["Ce texte permettra de montrer la possible troncature du padding si le nombre"
                            " de mot est inférieur à la taille maximum qu'on alloue à une phrase", "team3"],
                           ["encore une", "team2"]],
                          columns=["Text", "Team"])
        os.makedirs(os.path.dirname(fake_data_path), exist_ok=True)
        df.to_csv(fake_data_path, encoding='latin1')

    def get_data():
        from sklearn.model_selection import train_test_split
        from TextPreprocessing import TextPreprocessing
        df = pd.read_csv(fake_data_path, encoding='latin1')
        print(df)
        tp = TextPreprocessing(lemmatize=False)
        df.Text = tp.process(df.Text)
        data_train, data_test = train_test_split(df, test_size=TEST_RATIO)
        X_train = data_train.Text
        y_train = data_train.Team
        X_test = data_test.Text
        y_test = data_test.Team
        return (X_train, y_train), (X_test, y_test)

    if not os.path.exists(fake_data_path):
        create_fake_data()

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
    w2v_list = [True,]
    bidirectional = [True]

    # try with [0.4]
    dropout_list = [0.0]

    run_model = RunModel(X, y, X_test, y_test)

    for BATCH_SIZE in bs_list:
        for LAYERS in layers_list:
            for LEN_PADDING in len_pad_list:
                for LOSS in loss_list:
                    for OPTIMIZER in optimizer_list:
                        for LAYERS_TYPE in layers_type_list:
                            for DROPOUT in dropout_list:
                                for W2V in w2v_list:
                                    for BIDIRECTIONAL in bidirectional:
                                        run_model.model(BATCH_SIZE, LAYERS, LEN_PADDING,
                                                        LOSS, OPTIMIZER,LAYERS_TYPE, DROPOUT,
                                                        W2V, BIDIRECTIONAL)
