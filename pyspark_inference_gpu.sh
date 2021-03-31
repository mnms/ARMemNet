CUDF_JAR=/workspace/tfspark/cudf-0.18.1-cuda11.jar
PLUGIN_JAR=/workspace/tfspark/rapids-4-spark_2.12-0.4.1.jar

SPARK_HOME=/opt/spark-3.0.2-bin-hadoop3.2

rm -rf result

export PYTHONPATH=$PLUGIN_JAR

$SPARK_HOME/bin/spark-submit --master spark://127.0.0.1:7077            \
--jars ${CUDF_JAR},${PLUGIN_JAR}                       \
--conf spark.plugins=com.nvidia.spark.SQLPlugin        \
--driver-memory 10G                                    \
--executor-memory 30G                                  \
--conf spark.executor.cores=8 \
--conf spark.cores.max=65 \
--conf spark.rapids.memory.gpu.allocFraction=0.3       \
--conf spark.rapids.sql.python.gpu.enabled=true \
--conf spark.executor.extraClassPath=${CUDF_JAR}:${PLUGIN_JAR} \
--conf spark.executorEnv.PYTHONPATH=$PLUGIN_JAR \
--conf spark.driver.extraClassPath=${CUDF_JAR}:${PLUGIN_JAR} \
--conf spark.driver.extraJavaOptions="-Duser.timezone=UTC" \
--conf spark.rapids.sql.decimalType.enabled=true \
--conf spark.rapids.memory.pinnedPool.size=3g \
--conf spark.rapids.sql.hashOptimizeSort.enabled=true \
--conf spark.rapids.memory.gpu.pooling.enabled=true \
--conf spark.executor.extraJavaOptions="-Duser.timezone=UTC -Dai.rapids.cudf.prefer-pinned=true" \
--conf spark.driver.extraJavaOptions="-Duser.timezone=UTC" \
--conf spark.rapids.sql.expression.UnixTimestamp=true \
--conf spark.sql.parquet.outputTimestampType=TIMESTAMP_MICROS \
--conf spark.executor.resource.gpu.amount=1 \
--conf spark.rapids.sql.explain=ALL \
--conf spark.task.resource.gpu.amount=0.01 \
--conf spark.rapids.sql.concurrentGpuTasks=2 \
--conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
--files $SPARK_HOME/examples/src/main/scripts/getGpusResources.sh \
--py-files /workspace/ARMemNet/estimator_wrapper.py \
inference_mem_model_pyspark.py
