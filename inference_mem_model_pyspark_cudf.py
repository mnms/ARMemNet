import tensorflow as tf
import pandas as pd

from pyspark_utils import inference_data_as_pyspark_dataframe
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

model_path = "/workspace/ARMemNet/AR_mem/model_save/20210325-07"
data_path = "hdfs://192.168.201.250:9000/5min_parquet_inference"
result_path = "/workspace/ARMemNet/result"
data_format = "parquet"
inference_timestr = "20191218220000"
timestr_format = "%Y%m%d%H%M%S"

spark = SparkSession.builder.appName("ARMEM INFERENCE CUDF").getOrCreate()


def inference_cudf(pd_x: pd.Series, pd_m: pd.Series) -> pd.Series:
    import cudf

    model_path = '/workspace/tfspark/model'
    # make TF less agressive for GPU Memeory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # only one gpu visible to TF
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    model = tf.saved_model.load(model_path)

    gdf = cudf.DataFrame()
    # Since it's pd.Series to Series. I have to concat all columns. Liangcai's MapInPandas should help!
    gdf["0"] = cudf.Series(pd_x)
    gdf["1"] = cudf.Series(pd_m)

    tf_input_tsr = cudf_to_tensor(gdf)
    print(gdf.shape)

    # Hard code path, load model
    tf_out_tsr = model(tf_input_tsr)
    print("type of tf_out_tsr : " + str(type(tf_out_tsr)))
    return tf_out_tsr.to_pandas()


def cudf_to_tensor(gdf):
    import tensorflow as tf
    if gdf.empty:
        return
    # checks necessary because of this bug
    # https://github.com/tensorflow/tensorflow/issues/42660
    if len(gdf.shape) == 1 or gdf.shape[1] == 1:
        dlpack = gdf.to_dlpack()
    elif gdf.shape[0] == 1:
        dlpack = gdf.values[0].toDlpack()
    else:
        dlpack = gdf.values.T.toDlpack()

    # catch error caused by tf eager context
    # not being initialized
    try:
        x = tf.experimental.dlpack.from_dlpack(dlpack)
    except AssertionError:
        tf.random.uniform((1,))
        x = tf.experimental.dlpack.from_dlpack(dlpack)

    if gdf.shape[0] == 1:
        # batch size 1 so got squashed to a vector
        x = tf.expand_dims(x, 0)
    elif len(gdf.shape) == 1 or len(x.shape) == 1:
        # sort of a generic check for any other
        # len(shape)==1 case, could probably
        # be more specific
        x = tf.expand_dims(x, -1)
    elif gdf.shape[1] > 1:
        # matrix which means we had to transpose
        # for the bug above, so untranspose
        x = tf.transpose(x)
    return x


inference_udf = pandas_udf(inference_cudf, ArrayType(FloatType(), True))
input_df = spark.read.format(data_format).load(data_path)
df = inference_data_as_pyspark_dataframe(input_df, inference_timestr, timestr_format, inference_udf)

df.write.mode("overwrite").parquet(result_path)

check = spark.read.parquet(result_path)
check.show()
