#!/bin/bash

pip3 install -e '.[linting]'
python3 -m ruff check .
python3 -m pylint .