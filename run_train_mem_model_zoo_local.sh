#!/usr/bin/env bash
export OMP_NUM_THREADS=10

${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
 --master local[*] \
 --driver-memory 20g \
 train_mem_model_zoo.py /Users/wangjiao/data/skt 128 10 /Users/wangjiao/git/ARMemNet-jennie/model
