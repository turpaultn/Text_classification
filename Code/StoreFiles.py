import os
import json
import time
import random
import string
from Constantes_public import model_files_params, preprocessed_files_params
FILENAME = 'filename'

from Log import logger


# TODO use a Database to store document instead of manual method (example: MongoDB)
class StoreFiles:
    def __init__(self, file_type):
        self.file_type = file_type
        if file_type == 'model':
            self.filename_json = model_files_params
            self.extension = '.h5'
            self.mandatory_args = []

        elif file_type == 'preprocessing':
            self.filename_json = preprocessed_files_params
            self.extension = '.csv'
            self.mandatory_args = ["lemmatize"]
        else:
            raise NotImplementedError("Only implemented StoreFiles for 'model' or 'preprocessing' type")

    def load_json(self):
        if os.path.exists(self.filename_json):
            with open(self.filename_json, 'r') as f:
                params = json.load(f)
        else:
            logger.warn('File not found, new list')
            params = []
        return params

    def store_json(self, params):
        os.makedirs(os.path.dirname(self.filename_json), exist_ok=True)
        with open(self.filename_json, 'w') as f:
            json.dump(params, f, indent=4)

    def generate_name(self):
        fname = time.strftime("%d%m%y-%H%M%S")
        params = self.load_json()
        filenames = [dic[FILENAME] for dic in params]
        while fname+self.extension in filenames:
            fname += random.choice(string.ascii_letters)

        return fname + self.extension

    def verif_args(self, dic):
        for arg in self.mandatory_args:
            if arg not in dic:
                val = input("{} Write N to avoid adding this variable or type a value to store it".format(arg))
                val = val.strip()
                if not val == 'N':
                    logger.info('value {} added to var {}'.format(val, arg))
                    dic[arg] = val
        return dic

    def fname_and_store(self, type, **kwargs):
        fname, params_arranged = self.fname_to_store(type, **kwargs)
        self.store_json(params_arranged)
        return fname

    def fname_to_store(self, type, **kwargs):
        dic = {self.file_type: kwargs}
        dic["type"] = type
        dic = self.verif_args(dic)
        params = self.load_json()
        fname = self.generate_name()
        dic[FILENAME] = fname
        params.append(dic)
        return fname, params

    def get_fname(self, type=None, **kwargs):
        params = self.load_json()
        for dic in params:
            fname = dic.pop(FILENAME)
            if type is not None:
                if not dic["type"] == type:
                    continue

            if dic.get(self.file_type) == kwargs:
                return fname
        return None

    def get_params(self, fname):
        params = self.load_json()
        for dic in params:
            if fname == dic.get(FILENAME):
                return dic

    def update(self, filename, **kwargs):
        params = self.load_json()
        for dic in params:
            if dic[FILENAME] == filename:
                dic.update(kwargs)

    def remove(self, filename):
        params = self.load_json()
        for dic in params:
            if dic[FILENAME] == filename:
                logger.info("removed {}".format(filename))
                params.remove(dic)
        self.store_json(params)


if __name__ == '__main__':

    sf1 = StoreFiles('preprocessing')
    print(sf1.get_fname(dataset= "test", test_ratio= 0.2, random_test_samples= False,
                       lemmatized= False) == None)

    sf2 = StoreFiles('model')
    p = {'count':"bonjour", 'size':100}
    print(sf2.fname_and_store(type="word2vec", **p))

    sf3 = StoreFiles('model')
    print(sf3.get_fname(type="word2vec", **p))
    # sf = StoreFiles('model')
    # f = sf.store(layers=[256,128,56], type='LSTM', batch_size=32, optimizer='RMSprop')
    # print(f)