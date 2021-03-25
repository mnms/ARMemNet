from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import argparse
import tensorflow as tf

from tensorflowonspark import compat, TFNode
from tensorflowonspark import TFCluster
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from utils import get_logger, make_date_dir
from AR_mem.config import Config
from AR_mem.model import ARMemNet
from AR_mem.losses import RSE, SMAPE
from tensorflow.keras.losses import MSE, MAE
from tensorflow_estimator.python.estimator.export import export_lib
from pyspark_utils import *

def test_fn(config, ctx):
    logger, log_dir = get_logger(os.path.join(config.model, "logs/"))

    saved_model = tf.saved_model.load(config.test_model_dir, tags='serve')
    predict = saved_model.signatures['serving_default']

    tf_feed = TFNode.DataFeed(ctx.mgr, False)
    def rdd_generator():
        while not tf_feed.should_stop():
            batch = tf_feed.next_batch(1)
            if len(batch) > 0:
                data = batch[0]
                yield (data[0], data[1], data[2])
            else:
                return

    ds = tf.data.Dataset.from_generator(rdd_generator, (tf.float32, tf.float32, tf.float32), (tf.TensorShape([10, 8]), tf.TensorShape([8]), tf.TensorShape([77, 8])))
    ds = ds.shard(ctx.num_workers, ctx.worker_num)
    ds = ds.batch(config.batch_size)


    test_output_dir  = make_date_dir(os.path.join(config.model, 'test_output/'))
    test_output_file = tf.io.gfile.GFile("{}/worker={:05d}".format(test_output_dir,\
                                         ctx.worker_num), mode='w')
    
    for batch in ds:
        preds = predict((batch[0], batch[1]))
        labels = data[2]
        for x in zip(labels, preds):
            test_output_file.write("{} {} {}\n".format(x[0], x[1], x[2]))
    test_output.close()

if __name__ == '__main__':
    config = Config()

    sc        = SparkContext(conf=SparkConf().setAppName("ARmemNet_TF2onSpark_test"))
    spark     = SparkSession(sc)
    executors = sc._conf.get("spark.executor.instances")
    num_executors = int(executors) if executors is not None else 1
  
    raw_df = spark.read \
                  .schema(data_schema()) \
                  .csv("../data/aggregated_data_5min_scaled.csv", header=True, inferSchema=False)
    df_x, df_y, df_m = infernce_data_as_pyspark_dataframe(raw_df)
    ds_x = np.array(df_x.collect()).reshape(-1, 10, 8)
    ds_y = np.array(df_y.collect()).reshape(-1, 8)
    ds_m = np.array(df_m.collect()).reshape(-1, 77, 8)
    test_db = (ds_x, ds_m, ds_y)
  
    cluster = TFCluster.run(sc, test_fn, config, config.num_nodes, num_ps=0, tensorboard=config.use_tensorboard, input_mode=TFCluster.InputMode.SPARK, master_node='chief')
    # Note: need to feed extra data to ensure that each worker receives sufficient data to complete epochs
    # to compensate for variability in partition sizes and spark scheduling
    cluster.inference(test_db)
    cluster.shutdown()
