import json
import logging
import os
import pickle

import numpy as np
import spacy
import textacy
from gensim.models import KeyedVectors
from nltk import download
from nltk.corpus import stopwords

with open('settings.json') as file:
    config = json.load(file)

DATA_PATH = config.get('test-data-path')
D2V_VECTORS = DATA_PATH + '.d2v.vectors'
VECTORS_PATH = config.get('vectors-path')
CHUNK_SIZE = config.get('chunk-size', 1000)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.info('Loading stopwords...')
try:
    stopwords.words('english')
except LookupError:
    download('stopwords')
finally:
    STOPWORDS = set(stopwords.words('english'))
log.info('Stopwords were loaded...')
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


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def iter_text(filename):
    for _, doc in iter_id_with_text(filename):
        yield doc


def iter_id_with_text(filename):
    log.info('Loading spaCy model...')
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])
    log.info('SpaCy model was loaded...')
    with open(filename) as file:
        for index, article in enumerate(map(json.loads, file), 1):
            abstract = article.get('abstract', '')
            title = article.get('title', '')
            text = textacy.preprocess_text(title + '. ' + abstract, lowercase=True, transliterate=True, no_punct=True,
                                           no_numbers=True)
            terms_list = list(
                textacy.Doc(text, lang=nlp).to_terms_list(as_strings=True, named_entities=False, normalize='lemma',
                                                          ngrams=(1)))
            if index % CHUNK_SIZE == 0:
                log.info('%d articles were loaded...', index)
            id = article['id']
            yield id, [term for term in terms_list if term not in STOPWORDS]


if __name__ == "__main__":
    main()
