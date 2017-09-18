import re
from LemmatizeTreeTagger import LemmatizeTreeTagger
import pandas as pd

from Log import logger


class TextPreprocessing:
    def __init__(self, lemmatize=False, rm_num=True, rm_pun=True, lower=True):
        self.lemmatize = lemmatize
        self.rm_num = rm_num
        self.rm_pun = rm_pun
        self.lower = lower

    def clean_doc(self, document):
        cleaned_document = []
        splitted_document = re.split("\W", document)
        for word in splitted_document:
            if word != '' or word.isdigit():
                if self.lower:
                    word = word.lower()
                # if not word.isupper():
                # 	word = word.lower()
                cleaned_document.append(word)

        cleaned_document = ' '.join(cleaned_document)
        return cleaned_document

    def clean_docs(self, documents):
        cleaned_docs = []
        for i, document in enumerate(documents):
            cleaned_docs.append(self.clean_doc(document))
            if (i + 1) % 10000 == 0:
                logger.info("Review %d of %d" % (i+1, len(documents)))
        return cleaned_docs

    def process(self, documents):
        if self.lemmatize:
            docs = LemmatizeTreeTagger(rm_num=self.rm_num, rm_pun=self.rm_pun, lower=self.lower).lemmatize(documents)
        else:
            docs = self.clean_docs(documents)
        return docs


if __name__ == '__main__':
    Text_serie = pd.Series(["Bonjour, j'ai un problème ici", "Nous allons le réosudre d'ici là!!"])
    print(Text_serie)
    tp = TextPreprocessing(lemmatize=True)
    Text_serie = tp.process(Text_serie)
    print(Text_serie)