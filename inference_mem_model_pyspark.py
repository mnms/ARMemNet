import numpy as np
import pandas as pd
from estimator_wrapper import EstimatorWrapper

from pyspark_utils import inference_data_as_pyspark_dataframe
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

model_path = "/workspace/ARMemNet_TF2/AR_mem/model_save/20210325-07"
data_path = "hdfs://192.168.201.250:9000/5min_parquet_inference"
result_path = "/workspace/ARMemNet_TF2/result"
data_format = "parquet"
inference_timestr = "20191218220000"
timestr_format = "%Y%m%d%H%M%S"

spark = SparkSession.builder.appName("ARMEM INFERENCE").getOrCreate()

estimator_wrapper = EstimatorWrapper(model_path)
estimator_wrapper_broadcast = spark.sparkContext.broadcast(estimator_wrapper)


def inference(pd_x: pd.Series, pd_m: pd.Series) -> pd.Series:
    ret = pd.Series()
    for i in range(pd_x.size):
        x = np.array(pd_x[i]).reshape(-1, 10, 8)
        m = np.array(pd_m[i]).reshape(-1, 77, 8)
        outputs = estimator_wrapper_broadcast.value.estimator([x, m]).numpy().reshape(-1)
        ret = ret.append(pd.Series([outputs]))
    return ret


inference_udf = pandas_udf(inference, ArrayType(FloatType(), True))
input_df = spark.read.format(data_format).load(data_path)
df = inference_data_as_pyspark_dataframe(input_df, inference_timestr, timestr_format, inference_udf)

df.write.mode("overwrite").parquet(result_path)

check = spark.read.parquet(result_path)
check.select('out').show()
