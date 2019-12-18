export OMP_NUM_THREADS=10
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
 --master local[4] \
 --driver-memory 20g \
 train_mem_model_zoo.py /home/nvkvs/data/aggregated_5min_scaled.csv 128 10 /home/nvkvs/ARMemNet-jennie/model
