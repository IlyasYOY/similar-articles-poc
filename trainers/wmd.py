import json
import logging
import shelve

import networkx as nx
import progressbar
import spacy
import textacy
from gensim.models import KeyedVectors

with open('settings.json') as file:
    config = json.load(file)

DATA_PATH = config.get('test-citations-data-path')
GRAPH_DATA = DATA_PATH + '.graph.gml'
VECTORS_PATH = config.get('vectors-path')
CHUNK_SIZE = config.get('chunk-size', 1000)

log = logging.getLogger(__name__)
log.info('Loading model from: %s...', VECTORS_PATH)
model: KeyedVectors = KeyedVectors.load_word2vec_format(VECTORS_PATH)
model.init_sims(replace=True)
log.info('Model was loaded from: %s...', VECTORS_PATH)


def main():
    log.info('Loading spaCy model...')
    spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])
    log.info('SpaCy model was loaded...')
    graph = nx.MultiGraph()
    with open(DATA_PATH) as file, shelve.open('data/test.json.shelve') as db, progressbar.ProgressBar() as progress:
        for index, citation in enumerate(map(json.loads, file), 1):
            from_index = citation.get('from')
            from_article = db.get(str(from_index))
            from_text = preprocess_text(from_article)

            to_indexes = citation.get('to')
            to_articles = list(map(db.get, map(str, to_indexes)))
            to_articles_text = [preprocess_text(article) for article in to_articles]

            for citation_index, citation_text in zip(to_indexes, to_articles_text):
                dist = model.wmdistance(' '.join(from_text), ' '.join(to_articles_text))
                graph.add_edge(from_index, citation_index, distance=dist)
                progress.update(progress.value + 1)
    log.info("Writing graph....")
    nx.write_gml(graph, GRAPH_DATA)
    log.info("Graph was written....")


def split_to_terms(article_text, nlp):
    return list(
        textacy.Doc(article_text, lang=nlp).to_terms_list(as_strings=True, named_entities=False,
                                                          normalize='lemma', ngrams=(1)))


def preprocess_text(from_article):
    return textacy.preprocess_text(from_article.get('title', '') + '. ' + from_article.get('abstract', ''),
                                   lowercase=True, transliterate=True, no_punct=True,
                                   no_numbers=True)


if __name__ == '__main__':
    main()
