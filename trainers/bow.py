import json
import logging
import os
import pickle

import textacy

from trainers import iter_text

with open('settings.json') as file:
    config = json.load(file)

DATA_PATH = config.get('test-data-path')
BOW_VECTORIZER_PATH = DATA_PATH + '.bow.vectorizer'
BOW_VECTORS = DATA_PATH + '.bow.vectors'
CHUNK_SIZE = config.get('chunk-size', 1000)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    vectorizer = load_vecorizer()
    print(load_matrix(vectorizer))


def load_matrix(vectorizer):
    log.info('Loading matrix...')
    if not os.path.isfile(BOW_VECTORS):
        log.info('Creating whole new matrix...')
        docs = iter_text(DATA_PATH)
        matrix = vectorizer.transform(docs)
        log.info('Writing matrix...')
        with open(BOW_VECTORS, 'wb') as file:
            pickle.dump(matrix, file)
        log.info('Matrix was written...')
    else:
        log.info('Loadign matrix from disk...')
        with open(BOW_VECTORS, 'rb') as file:
            matrix = pickle.load(file)
        log.info('matrix was loaded...')
    return matrix


def load_vecorizer():
    if not os.path.isfile(BOW_VECTORIZER_PATH):
        vectorizer = textacy.Vectorizer(min_df=2, max_df=0.95, norm='l2')
        with open(BOW_VECTORIZER_PATH, 'wb') as file:
            log.info('Training vectorizer on data from %s...', DATA_PATH)
            docs = iter_text(DATA_PATH)
            vectorizer = vectorizer.fit(docs)
            log.info('Vectorizer was trained, writing it here %s...',
                     BOW_VECTORIZER_PATH)
            pickle.dump(vectorizer, file)
    else:
        with open(BOW_VECTORIZER_PATH, 'rb') as file:
            log.info('Loading vectorizer from %s', BOW_VECTORIZER_PATH)
            vectorizer = pickle.load(file)
            log.info('Vectorizer was loaded from %s', BOW_VECTORIZER_PATH)
    return vectorizer


if __name__ == '__main__':
    main()
