from zoo import init_nncontext
from zoo.tfpark import TFOptimizer, TFDataset
from bigdl.optim.optimizer import *
from data_utils import load_agg_selected_data_mem_train
from AR_mem.config import Config
from AR_mem.model import Model
from time import time
import tensorflow as tf
from zoo.common import set_core_number


if __name__ == "__main__":

    data_path = sys.argv[1]
    batch_size = int(sys.argv[2])
    num_epochs = int(sys.argv[3])
    model_dir = sys.argv[4]
    if len(sys.argv) > 5:
        core_num = sys.argv[5]
    else:
        core_num = 4
    if len(sys.argv) > 6:
        thread_num = sys.argv[6]
    else:
        thread_num = 10

    config = Config()
    config.data_path = data_path
    config.latest_model=False

    # init or get SparkContext
    sc = init_nncontext()
    set_core_number(core_num)

    # create train data
    train_x, dev_x, test_x, train_y, dev_y, test_y, train_m, dev_m, test_m, test_dt = \
        load_agg_selected_data_mem_train(data_path=config.data_path,
                                   x_len=config.x_len,
                                   y_len=config.y_len,
                                   foresight=config.foresight,
                                   cell_ids=config.train_cell_ids,
                                   dev_ratio=config.dev_ratio,
                                   test_len=config.test_len,
                                   seed=config.seed)

    dataset = TFDataset.from_ndarrays([train_x, train_m, train_y], batch_size=batch_size, val_tensors=[dev_x, dev_m, dev_y],)

    model = Model(config, dataset.tensors[0], dataset.tensors[1], dataset.tensors[2])
    optimizer = TFOptimizer.from_loss(model.loss, Adam(config.lr),
                                      metrics={"rse": model.rse, "smape": model.smape, "mae": model.mae},
                                      model_dir=model_dir,
                                      session_config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                                                    intra_op_parallelism_threads=thread_num)
                                      )

    start_time = time()
    optimizer.optimize(end_trigger=MaxEpoch(num_epochs))
    end_time = time()

    print("Elapsed training time {} secs".format(end_time - start_time))


