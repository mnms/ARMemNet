import datetime
from datetime import datetime, timedelta

from pyspark.sql import Window
from pyspark.sql.functions import *
from pyspark.sql.types import *


def data_schema():
    return StructType() \
        .add("EVT_DTM", StringType(), True) \
        .add("CQI", FloatType(), True) \
        .add("RSRP", FloatType(), True) \
        .add("RSRQ", FloatType(), True) \
        .add("DL_PRB_USAGE_RATE", FloatType(), True) \
        .add("SINR", FloatType(), True) \
        .add("UE_TX_POWER", FloatType(), True) \
        .add("PHR", FloatType(), True) \
        .add("UE_CONN_TOT_CNT", FloatType(), True) \
        .add("CELL_NUM", IntegerType(), True)


def features():
    return ["CQI", "RSRP", "RSRQ", "DL_PRB_USAGE_RATE", "SINR", "UE_TX_POWER", "PHR", "UE_CONN_TOT_CNT"]


def training_data_as_pyspark_dataframe(
        input_df,
        feature=features(),
        day_gap=13 * 60 // 5,
        x_len=10,
        mem_len=7
):
    df = input_df
    win_spec = Window().partitionBy(col("CELL_NUM")).orderBy(desc("EVT_DTM"))
    df = df.withColumn("FEATURES", array(feature))
    feature_cols = ["FEATURES"]
    for i in range(1, x_len):
        df = df.withColumn("FEATURES-" + str(i), lead(col("FEATURES"), i).over(win_spec))
        feature_cols.append("FEATURES-" + str(i))
    df = df.withColumn("X", array(feature_cols)) \
        .withColumn("Y", lag(col("FEATURES"), 1).over(win_spec)) \
        .dropna() \
        .select("EVT_DTM", "CELL_NUM", "X", "Y")

    for i in range(1, mem_len + 1):
        df = df.withColumn("M-" + str(i), lead(concat(array("Y"), "X"), day_gap * i).over(win_spec))

    df = df.dropna() \
        .withColumn("M", concat(*list(map(lambda i: "M-" + str(i), range(1, mem_len + 1))))) \
        .select("EVT_DTM", "CELL_NUM", "X", "Y", "M")
    df = df.cache()
    return df.select("X"), df.select("Y"), df.select("M")


def inference_data_as_pyspark_dataframe(
        input_df,
        input_datetime_str,
        dateformat_str,
        x_len=10,
        mem_len=7
):
    def buildMemoryTimeList(d, dateformat_str):
        output = []
        for day in range(-1, -(mem_len + 1), -1):
            for min in range(5, -(5 * x_len), -5):
                delta = timedelta(days=day, minutes=min)
                output.append((d + delta).strftime(dateformat_str))
        return output

    def listToString(l):
        if len(l) == 0:
            return ""
        str = ""
        for i in range(0, len(l) - 1):
            str += "'{}',".format(l[i])
        str += "'{}'".format(l[len(l) - 1])
        return str

    input_datetime = datetime.strptime(input_datetime_str, dateformat_str)
    memory_times = buildMemoryTimeList(input_datetime, dateformat_str)
    time_filter = "evt_dtm IN ({})".format(listToString(memory_times))
    inf_df_memory = input_df.where(time_filter) \
        .groupBy("cell_num").pivot("evt_dtm") \
        .agg(array(first("CQI"), first("RSRP"), first("RSRQ"), first("DL_PRB_USAGE_RATE"), first("SINR"),
                   first("UE_TX_POWER"), first("PHR"), first("UE_CONN_TOT_CNT"))) \
        .select("cell_num", array(memory_times).alias("M"))
    inf_df_x = input_df.where("evt_dtm == '{}'".format(input_datetime_str)) \
        .select("cell_num", array("CQI", "RSRP", "RSRQ", "DL_PRB_USAGE_RATE", "SINR", "UE_TX_POWER", "PHR",
                                  "UE_CONN_TOT_CNT").alias("X"))
    inf_df = inf_df_memory.join(inf_df_x, on='cell_num', how='left')
    return inf_df


# Fill every time slot in time bound, and fill null values in features with front-fill & back-fill
def prepare_inference_data(
        spark,
        raw_data_df,
        earliest_evt_dtm,
        evt_dtm_count
):
    def build_time_list():
        output = []
        start_time = datetime.strptime(earliest_evt_dtm, "%Y%m%d%H%M%S")
        for i in range(0, evt_dtm_count):
            delta = timedelta(minutes=5 * i)
            output.append((start_time + delta).strftime("%Y%m%d%H%M%S"))
        return output

    timelist = build_time_list()
    df_time = spark.createDataFrame(timelist, StringType()).withColumnRenamed("value", "evt_dtm")
    df_cell = raw_data_df.withColumn("cell_num", (col("enb_id") * 100 + col("cell_id"))).select("cell_num").distinct()
    df_time_cell = df_cell.crossJoin(df_time)

    df_5min = raw_data_df.withColumn("cell_num", (col("enb_id") * 100 + col("cell_id"))).select(
        ["evt_dtm", "cell_num"] + features())
    df_5min = df_time_cell.join(df_5min, on=["evt_dtm", "cell_num"], how="left")

    window_ff = Window().partitionBy(col("cell_num")).orderBy("evt_dtm").rowsBetween(-sys.maxsize, 0)
    window_bf = Window().partitionBy(col("cell_num")).orderBy("evt_dtm").rowsBetween(0, sys.maxsize)

    for feature in features():
        read_last = last(df_5min[feature], ignorenulls=True).over(window_ff)
        read_next = first(df_5min[feature], ignorenulls=True).over(window_bf)
        df_5min = df_5min.withColumn(feature + "_ff", read_last).withColumn(feature, read_next).drop(feature + "_ff")
    df_5min = df_5min.dropna()

    return df_5min
