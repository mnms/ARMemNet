from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from pyspark.context import SparkContext
from pyspark.conf import SparkConf
import tensorflow as tf
import tensorflow_datasets as tfds

def main():
  
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_partitions", help="Number of output partitions", type=int, default=10)
    parser.add_argument("--output", help="HDFS directory to save examples in parallelized format", default="data/armem")
  
    args = parser.parse_args()
    print("args:", args)
  
    sc = SparkContext(conf=SparkConf().setAppName("armemn_data_setup"))
  
    armem_data, info = tfds.load('armem', with_info=True)
    print(info.as_json)
  
    # convert to numpy, then RDDs
    armem_train = tfds.as_numpy(armem_data['train'])
    armem_test = tfds.as_numpy(armem_data['test'])
  
    train_rdd = sc.parallelize(armem_train, args.num_partitions).cache()
    test_rdd = sc.parallelize(armem_test, args.num_partitions).cache()

    # save as CSV (label,comma-separated-features)
    def to_csv(data):
        return str(data['label']) + ',' + ','.join([str(i) for i in data['X'].flatten()]) + ','\
                + ','.join([str(i) for i in data['mem'].flatten()])

    train_rdd.map(to_csv).saveAsTextFile(args.output + "/csv/train")
    test_rdd.map(to_csv).saveAsTextFile(args.output + "/csv/test")
  
if __name__ == "__main__":
    main()
