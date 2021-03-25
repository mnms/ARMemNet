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

def train_fn(config, ctx):
    logger, log_dir = get_logger(os.path.join(config.model, "logs/"))

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

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
    ds = ds.batch(config.batch_size)

    checkpoint_dir    = make_date_dir(os.path.join(config.model, 'ckpt_save/'))
    checkpoint_prefix = os.path.join(checkpoint_dir, config.model)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience = config.nepoch_no_improv, monitor='val_loss'), \
        tf.keras.callbacks.TensorBoard(log_dir = make_date_dir(os.path.join(config.model, 'model_log/'))), \
        tf.keras.callbacks.ModelCheckpoint( filepath = checkpoint_prefix, \
                                            verbose=1, \
                                            save_weights_only=True, \
                                            mode='max', \
                                            monitor='val_loss' )]

    with strategy.scope():
        multi_worker_model = ARMemNet(config)
        multi_worker_model.compile( \
            loss      = MSE, \
            optimizer = tf.keras.optimizers.Adam(learning_rate=(config.lr)), \
            metrics   = [MSE, RSE, SMAPE, MAE])

    # Note: MultiWorkerMirroredStrategy (CollectiveAllReduceStrategy) is synchronous,
    # so we need to ensure that all workers complete training before any of them run out of data from the RDD.
    # And given that Spark RDD partitions (and partition sizes) can be non-evenly divisible by num_workers,
    # we'll just stop training at 90% of the total expected number of steps.
    steps_per_epoch            = int( config.num_data / config.batch_size )
    steps_per_epoch_per_worker = int( steps_per_epoch / ctx.num_workers )
    max_steps_per_worker       = int( steps_per_epoch_per_worker * 0.9 )

    # [TODO] max_queue_size(default=10), need to fit
    multi_worker_model.fit( verbose             = 2, \
                            x                   = ds, \
                            epochs              = config.num_epochs, \
                            steps_per_epoch     = max_steps_per_worker, \
                            validation_data     = 0.3, \
                            max_queue_size      = config.max_queue_size,\
                            callbacks           = callbacks, \
                            use_multiprocessing = True )

    export_dir = export_lib.get_timestamped_export_dir(os.path.join(config.model, 'model_save/'))
    compat.export_saved_model(multi_worker_model, export_dir, ctx.job_name == 'chief')

    # terminating feed tells spark to skip processing further partitions
    tf_feed.terminate()

if __name__ == '__main__':
    config = Config()

    sc        = SparkContext(conf=SparkConf().setAppName("ARmemNet_TF2onSpark_train"))
    spark     = SparkSession(sc)
    executors = sc._conf.get("spark.executor.instances")
    num_executors = int(executors) if executors is not None else 1
  
    raw_df = spark.read \
                  .schema(data_schema()) \
                  .csv("../data/aggregated_data_5min_scaled.csv", header=True, inferSchema=False)
    df_x, df_y, df_m = training_data_as_pyspark_dataframe(raw_df)
    ds_x = np.array(df_x.collect()).reshape(-1, 10, 8)
    ds_y = np.array(df_y.collect()).reshape(-1, 8)
    ds_m = np.array(df_m.collect()).reshape(-1, 77, 8)
    train_db = (ds_x, ds_m, ds_y)
  
    cluster = TFCluster.run(sc, train_fn, config, config.num_nodes, num_ps=0, tensorboard=config.use_tensorboard, input_mode=TFCluster.InputMode.SPARK, master_node='chief')
    # Note: need to feed extra data to ensure that each worker receives sufficient data to complete epochs
    # to compensate for variability in partition sizes and spark scheduling
    cluster.train(train_db, config.num_epochs)
    cluster.shutdown()
