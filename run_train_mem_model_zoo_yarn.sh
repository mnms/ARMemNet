#!/usr/bin/env bash
export PYSPARK_PYTHON=python3
export OMP_NUM_THREADS=10
export HADOOP_CONF_DIR=${HADOOP_HOME}/etc/hadoop

if [ -f train.zip ]
  then rm train.zip
fi
zip -r train.zip *.py */*.py

${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
 --master yarn \
 --deploy-mode client \
 --driver-memory 6g \
 --executor-memory 6g \
 --executor-cores 2 \
 --num-executors 1 \
 --py-files train.zip \
 train_mem_model_zoo.py hdfs://localhost:9000/skt 128 10 /Users/wangjiao/git/ARMemNet-jennie/model
