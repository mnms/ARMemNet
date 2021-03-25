#!/bin/bash

export TFoS_HOME=/root/git/TensorFlowOnSpark
export MASTER=spark://localhost:8888
export SPARK_WORKER_INSTANCES=8
export CORES_PER_WORKER=2
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
export ARMemNET_TF2onSpark=/workspace/model/ARMemNet_TF2onSpark

${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-slave.sh -c $CORES_PER_WORKER -m 3G ${MASTER}

# confirm that data was generated
ls -lR ../data/aggregated_data_5min_scaled.csv

${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
${ARMemNET_TF2onSpark}/armemnet_test.py 
