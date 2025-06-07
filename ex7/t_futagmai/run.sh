#!/bin/bash
# セットアップで作成した自分のvirtualenv環境をロード
source ${HOME}/workspace2/b4lecture-2025/.venv/bin/activate

# CUDA_VISIBLE_DEVICESを空に設定
export CUDA_VISIBLE_DEVICES=""

# 作成したPythonスクリプトを実行
python -u main.py --path_to_truth ../test_truth.csv