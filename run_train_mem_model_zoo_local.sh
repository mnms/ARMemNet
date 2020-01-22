#!/usr/bin/env bash
export PYSPARK_PYTHON=python3
export OMP_NUM_THREADS=10

${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
 --master local[2] \
 --driver-memory 20g \
 train_mem_model_zoo.py hdfs://localhost:9000/skt 128 10 /Users/wangjiao/git/ARMemNet-jennie/model
