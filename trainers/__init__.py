import json
import logging

import numpy as np
import spacy
import textacy
from nltk import download
from nltk.corpus import stopwords

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


def iter_id_with_text(filename, chunk_size=1000):
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
            if index % chunk_size == 0:
                log.info('%d articles were loaded...', index)
            id = article['id']
            yield id, [term for term in terms_list if term not in STOPWORDS]


def iter_text(filename):
    for _, doc in iter_id_with_text(filename):
        yield doc


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
