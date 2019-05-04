import json
import logging
import os
import pickle

import textacy

from trainers import iter_text

with open('settings.json') as file:
    config = json.load(file)

DATA_PATH = config.get('test-data-path')
TFIDF_VECTORIZER_PATH = DATA_PATH + '.tfidf.vectorizer'
TFIDF_VECTORS = DATA_PATH + '.tfidf.vectors'
CHUNK_SIZE = config.get('chunk-size', 1000)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    vectorizer = load_vecorizer()
    print(load_matrix(vectorizer))


def load_matrix(vectorizer):
    log.info('Loading matrix...')
    if not os.path.isfile(TFIDF_VECTORS):
        log.info('Creating whole new matrix...')
        docs = iter_text(DATA_PATH)
        matrix = vectorizer.transform(docs)
        log.info('Writing matrix...')
        with open(TFIDF_VECTORS, 'wb') as file:
            pickle.dump(matrix, file)
        log.info('Matrix was written...')
    else:
        log.info('Loadign matrix from disk...')
        with open(TFIDF_VECTORS, 'rb') as file:
            matrix = pickle.load(file)
        log.info('matrix was loaded...')
    return matrix


def load_vecorizer():
    if not os.path.isfile(TFIDF_VECTORIZER_PATH):
        vectorizer = textacy.Vectorizer(
            apply_idf=True, min_df=2, max_df=0.95, norm='l2')
        with open(TFIDF_VECTORIZER_PATH, 'wb') as file:
            log.info('Training vectorizer on data from %s...', DATA_PATH)
            docs = iter_text(DATA_PATH)
            vectorizer = vectorizer.fit(docs)
            log.info('Vectorizer was trained, writing it here %s...',
                     TFIDF_VECTORIZER_PATH)
            pickle.dump(vectorizer, file)
    else:
        with open(TFIDF_VECTORIZER_PATH, 'rb') as file:
            log.info('Loading vectorizer from %s', TFIDF_VECTORIZER_PATH)
            vectorizer = pickle.load(file)
            log.info('Vectorizer was loaded from %s', TFIDF_VECTORIZER_PATH)
    return vectorizer


if __name__ == '__main__':
    main()
