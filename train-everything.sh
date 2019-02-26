#!/usr/bin/env bash

python -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt

echo "Running bow.py..."
./.venv/bin/python trainers/bow.py

echo "Running tfidf.py..."
./.venv/bin/python trainers/tfidf.py

echo "Running d2v.py..."
./.venv/bin/python trainers/d2v.py

echo "Running shelver.py..."
./.venv/bin/python trainers/shelver.py

echo "Running wmd.py"
./.venv/bin/python trainers\wmd.py
