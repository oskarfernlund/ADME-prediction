#!/bin/bash
pip install --upgrade pip
pip install -r requirements.txt
pip install git+https://github.com/pyg-team/pyg-lib.git
pip install torch-scatter
pip install torch-sparse
pip install torch-geometric
