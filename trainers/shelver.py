import json
import logging
import os
import shelve

with open('settings.json') as file:
    config = json.load(file)

DATA_PATH = config.get('test-data-path')
SHELVE_PATH = DATA_PATH + '.shelve'
CHUNK_SIZE = config.get('chunk-size', 1000)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    if not os.path.isfile(SHELVE_PATH):
        log.info("Creating a new shelve...")
        with shelve.open(SHELVE_PATH) as db:
            for index, article in iter_index_with_article(DATA_PATH):
                db[index] = article
    else:
        log.info('Shelve already exists...')


def iter_index_with_article(filename):
    with open(filename) as file:
        for index, article in enumerate(map(json.loads, file), 1):
            if index % CHUNK_SIZE == 0:
                log.info('%d articles were loaded...', index)
            yield str(article['id']), article


if __name__ == '__main__':
    main()
