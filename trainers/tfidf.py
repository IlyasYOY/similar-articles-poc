import json
import logging
import os
import pickle

import spacy
import textacy
from nltk import download
from nltk.corpus import stopwords

with open('settings.json') as file:
    config = json.load(file)

DATA_PATH = config.get('test-data-path')
TFIDF_VECTORIZER_PATH = DATA_PATH + '.tfidf.vectorizer'
TFIDF_VECTORS = DATA_PATH + '.tfidf.vectors'
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


def iter_text(filename):
    for _, doc in iter_id_with_text(filename):
        yield doc


def iter_id_with_text(filename):
    log.info('Loading spaCy model...')
    nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])
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


if __name__ == '__main__':
    main()
