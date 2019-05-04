import json
import logging
import os
import pickle

import numpy as np
from gensim.models import KeyedVectors

from trainers import iter_text, normalize

with open('settings.json') as file:
    config = json.load(file)

DATA_PATH = config.get('test-data-path')
D2V_VECTORS = DATA_PATH + '.d2v.vectors'
VECTORS_PATH = config.get('vectors-path')
CHUNK_SIZE = config.get('chunk-size', 1000)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.info('Loading model from: %s...', VECTORS_PATH)
model: KeyedVectors = KeyedVectors.load_word2vec_format(VECTORS_PATH)
model.init_sims(replace=True)
log.info('Model was loaded from: %s...', VECTORS_PATH)


def main():
    if not os.path.isfile(D2V_VECTORS):
        log.info('Building whole new matrix...')
        matrix = build_matrix()
        log.info('Saving freshly built matrix...')
        with open(D2V_VECTORS, 'wb') as file:
            pickle.dump(matrix, file)
        log.info('Matrix was saved...')
    else:
        log.info('Reading matrix from file...')
        with open(D2V_VECTORS, 'rb') as file:
            matrix = pickle.load(file)
        log.info('Matrix was read from file...')


def build_matrix():
    vectors = []
    for text in iter_text(DATA_PATH):
        text_vector = np.zeros((1, model.vector_size))
        for word in text:
            try:
                vector = model.get_vector(word)
            except KeyError:
                continue
            text_vector += vector
        vectors.append(normalize(text_vector))
    return np.array(vectors).reshape((-1, 300))


if __name__ == "__main__":
    main()
