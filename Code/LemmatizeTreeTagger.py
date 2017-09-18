import time
import pandas as pd
import treetaggerpoll
import os


import logging
from Log import logger

def root_path():
    return os.path.abspath(os.sep)


class LemmatizeTreeTagger:
    def __init__(self, nprocessors=None, TAGLANG="fr", TAGDIR=os.path.join(root_path(), 'TreeTagger'), rm_num=True,
                 rm_pun=True, lower=True, **kwargs):
        self.tagger = treetaggerpoll.TaggerProcessPoll(workerscount=nprocessors,
                                                       TAGLANG=TAGLANG, TAGDIR=TAGDIR, **kwargs)
        self.rm_num = rm_num
        self.rm_pun = rm_pun
        self.lower = lower

    def get_words(self, tag):
        if tag == '':
            return tag

        tag_splitted = tag.split('\t')
        if len(tag_splitted) == 1:  # special characters
            return ''
        if self.rm_num:
            if 'NUM' in tag_splitted[1]:
                return ''
        if self.rm_pun:
            if 'PUN' in tag_splitted[1] or 'SENT' in tag_splitted[1]:
                return ''
        if self.lower:
            return tag_splitted[-1].lower()
        return tag_splitted[-1]

    def lemmatize(self, texts):
        logging.debug('in the function')
        start = time.time()

        res = []

        lem_texts = []

        logger.info("Creating jobs")
        for i in range(len(texts)):
            # logger.info("   Job", i)
            res.append(self.tagger.tag_text_async(texts[i]))

        logger.info("Waiting for jobs to complete")
        for i, tags in enumerate(res):
            tags.wait_finished()
            lem_texts.append(' '.join(list(filter(None, map(self.get_words, tags.result)))))
            res[i] = None   # Loose Job reference - free it.

        self.tagger.stop_poll()
        logger.info("Finished after {:0.2f} seconds elapsed".format(time.time() - start))
        return lem_texts


if __name__ == '__main__':
    Text_serie = pd.Series(["bonjour j'ai un problème ici",
                            "nous allons le réosudre d'ici là",
                            "nous voulons l'améliorer",
                            "je suis maintenant devant des problèmes"])
    print(Text_serie)
    lemmatized_serie = LemmatizeTreeTagger().lemmatize(Text_serie)
    print(lemmatized_serie)