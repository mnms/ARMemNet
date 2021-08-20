#!/bin/bash
export SPARK_RAPIDS_DIR=$HOME/sparkRapidsPlugin
export SPARK_HOME=$HOME/spark
#export SPARK_RAPIDS_PLUGIN_JAR=$SPARK_RAPIDS_DIR/rapids-4-spark_2.12-21.08.0.jar
#export SPARK_CUDF_JAR=$SPARK_RAPIDS_DIR/cudf-21.08.2-cuda11.jar
export SPARK_RAPIDS_PLUGIN_JAR=$SPARK_RAPIDS_DIR/rapids-4-spark_2.12-21.06.2.jar
export SPARK_CUDF_JAR=$SPARK_RAPIDS_DIR/cudf-21.06.1-cuda11.jar
export SPARK_MASTER=spark://a205:7077

#rm -rf result
export PYSPARK_PYTHON=${HOME}/miniconda3/envs/py3.7/bin/python
export PYSPARK_DRIVER_PYTHON=${HOME}/miniconda3/envs/py3.7/bin/python
export PYTHONPATH=$SPARK_RAPIDS_PLUGIN_JAR
export CUDA_VISIBLE_DEVICES=""

export RAPIDS_SHUFFLE_CONFS="--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark311.RapidsShuffleManager \
        --conf spark.shuffle.service.enabled=false \
        --conf spark.executorEnv.UCX_TLS=cuda_copy,cuda_ipc,rc,tcp \
        --conf spark.executorEnv.UCX_ERROR_SIGNALS= \
        --conf spark.executorEnv.UCX_RNDV_SCHEME=put_zcopy \
        --conf spark.executorEnv.UCX_MAX_RNDV_RAILS=1 \
        --conf spark.executorEnv.UCX_MEMTYPE_CACHE=n \
        --conf spark.executorEnv.UCX_IB_RX_QUEUE_LEN=1024 \
        --conf spark.executor.extraClassPath=${SPARK_CUDF_JAR}:${SPARK_RAPIDS_PLUGIN_JAR}"

$SPARK_HOME/bin/spark-submit --master ${SPARK_MASTER}            \
--jars ${SPARK_CUDF_JAR},${SPARK_RAPIDS_PLUGIN_JAR}                       \
--conf spark.plugins=com.nvidia.spark.SQLPlugin        \
--driver-memory 10G                                    \
--executor-memory 30G                                  \
--conf spark.executor.cores=32 \
--conf spark.executor.instances=6 \
--conf spark.rapids.memory.gpu.allocFraction=0.3       \
--conf spark.rapids.sql.python.gpu.enabled=true \
--conf spark.rapids.memory.gpu.pooling.enabled=false \
--conf spark.rapids.python.memory.gpu.pooling.enabled=false \
--conf spark.executor.extraClassPath=${SPARK_CUDF_JAR}:${SPARK_RAPIDS_PLUGIN_JAR} \
--conf spark.executorEnv.PYTHONPATH=${SPARK_RAPIDS_PLUGIN_JAR} \
--conf spark.executorEnv.PYSPARK_PYTHON=${PYSPARK_PYTHON} \
--conf spark.driver.extraClassPath=${SPARK_CUDF_JAR}:${SPARK_RAPIDS_PLUGIN_JAR} \
--conf spark.driver.extraJavaOptions="-Duser.timezone=UTC" \
--conf spark.rapids.sql.decimalType.enabled=true \
--conf spark.rapids.memory.pinnedPool.size=0 \
--conf spark.rapids.sql.hashOptimizeSort.enabled=false \
--conf spark.rapids.memory.gpu.pooling.enabled=true \
--conf spark.executor.extraJavaOptions="-Duser.timezone=UTC -Dai.rapids.cudf.prefer-pinned=true" \
--conf spark.driver.extraJavaOptions="-Duser.timezone=UTC" \
--conf spark.rapids.sql.expression.UnixTimestamp=true \
--conf spark.sql.parquet.outputTimestampType=TIMESTAMP_MICROS \
--conf spark.executor.resource.gpu.amount=1 \
--conf spark.rapids.sql.explain=ALL \
--conf spark.task.resource.gpu.amount=0.03125 \
--conf spark.rapids.sql.concurrentGpuTasks=2 \
--conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
--files $SPARK_HOME/examples/src/main/scripts/getGpusResources.sh \
$RAPIDS_SHUFFLE_CONFS \
inference_mem_model_pyspark.py
