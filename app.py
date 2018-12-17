import functools
import json
import logging
import math
import pickle
import shelve
import os

import numpy
from flask import Flask, redirect, render_template, request, url_for, send_from_directory
from scipy.sparse import csr_matrix

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
app = Flask(__name__)
with open('settings.json') as file:
    config = json.load(file)

DOC2VEC_DATA = config.get('test-data-path') + '.d2v.vectors'
BOW_DATA = config.get('test-data-path') + '.bow.vectors'
TFIDF_DATA = config.get('test-data-path') + '.tfidf.vectors'
SHELVE_DATA = config.get('test-data-path') + '.shelve'

with open(DOC2VEC_DATA, 'rb') as file:
    log.info('Loading Doc2vec data from %s...', DOC2VEC_DATA)
    app.doc2vec: csr_matrix = csr_matrix(pickle.load(file))
    log.info('Doc2vec data were loaded...')
with open(BOW_DATA, 'rb') as file:
    log.info('Loading BoW data from %s...', BOW_DATA)
    app.bow: csr_matrix = pickle.load(file)
    log.info('BoW data were loaded...')
with open(TFIDF_DATA, 'rb') as file:
    log.info('Loading TF-IDF data from %s...', TFIDF_DATA)
    app.tfidf: csr_matrix = pickle.load(file)
    log.info('TF-IDF data were loaded...')


@app.route('/')
def index():
    from_index = int(request.args.get('from', 1))
    articles = []
    with shelve.open(SHELVE_DATA) as db:
        for index in sorted(map(int, list(db.keys()))):
            article = db[str(index)]
            if index >= from_index:
                articles.append(article)
            if len(articles) > 10:
                break
    return render_template('index.html', articles=articles, from_index=from_index)


@app.route('/recommend/<int:target>')
def recommend(target: int):
    limit = int(request.args.get('limit', 10))
    algorithm = request.args.get('algorithm').lower()
    index_score_pairs = recommend_with_algorithm(target, limit, algorithm)
    with shelve.open(SHELVE_DATA) as db:
        recommendations = [{'article': db.get(str(index)), 'score': "{:.2f}".format(
            score * 100)} for index, score in index_score_pairs]
        target_article = recommendations[0]['article']
    return render_template('results.html', target=target, limit=limit, recommendations=recommendations[1:],
                           algorithm=algorithm.upper(), target_article=target_article)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

    
@functools.lru_cache(maxsize=10)
def recommend_with_algorithm(target, limit, algorithm):
    if algorithm == 'tfidf':
        table = app.tfidf
    elif algorithm == 'bow':
        table = app.bow
    elif algorithm == 'doc2vec':
        table = app.doc2vec
    else:
        return None
    return recommend_algorithm(table, target, limit)


def recommend_algorithm(table, target, limit):
    vector = table.getrow(target)
    dot_column = table.dot(vector.T)
    recommendations = sorted(enumerate((item.data[0] if item.size > 0 else 0 for item in dot_column)),
                             key=lambda x: x[1], reverse=True)
    return recommendations[:limit + 1]


def main():
    app.run()


if __name__ == "__main__":
    main()
