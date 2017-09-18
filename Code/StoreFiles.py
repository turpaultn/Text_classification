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

    def fname_and_store(self, params):
        fname, params = self.fname_to_store(params)
        self.store_json(params)
        return fname

    def fname_to_store(self, params):
        assert type(params) is dict, "parameters to store must be a dictionary"
        dic = {self.file_type: params}
        dic = self.verif_args(dic)
        params = self.load_json()
        fname = self.generate_name()
        dic[FILENAME] = fname
        params.append(dic)
        return fname, params

    def get_fname(self, **kwargs):
        params = self.load_json()
        for dic in params:
            fname = dic.pop(FILENAME)
            if dic == kwargs:
                return fname
        return None

    def get_params(self, fname):
        params = self.load_json()
        for dic in params:
            if fname == dic.get(FILENAME):
                return dic

    def get_fname_by_part_of_params(self, params):
        models_params = self.load_json()
        key = self.file_type
        for dic in models_params:
            if dic.get(key) == params:
                return dic.get(FILENAME)
        return None

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

    sf = StoreFiles('preprocessing')
    print(sf.get_fname(dataset= "test", test_ratio= 0.2, random_test_samples= False,
                       lemmatized= False))
    # sf = StoreFiles('model')
    # f = sf.store(layers=[256,128,56], type='LSTM', batch_size=32, optimizer='RMSprop')
    # print(f)