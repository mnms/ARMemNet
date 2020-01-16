from zoo import init_nncontext
from zoo.tfpark import TFOptimizer, TFDataset
from bigdl.optim.optimizer import *
from data_utils import load_agg_selected_data_mem_train, get_datasets_from_dir, get_datasets_from_dir_spark
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

    # For tuning
    if len(sys.argv) > 5:
        core_num = int(sys.argv[5])
    else:
        core_num = 4
    if len(sys.argv) > 6:
        thread_num = int(sys.argv[6])
    else:
        thread_num = 10

    config = Config()
    config.data_path = data_path
    config.latest_model=False
    config.batch_size = batch_size

    # init or get SparkContext
    sc = init_nncontext()
    
    # tuning
    set_core_number(core_num)

    # create train data
    # train_x, dev_x, test_x, train_y, dev_y, test_y, train_m, dev_m, test_m, test_dt = \
    #     load_agg_selected_data_mem_train(data_path=config.data_path,
    #                                x_len=config.x_len,
    #                                y_len=config.y_len,
    #                                foresight=config.foresight,
    #                                cell_ids=config.train_cell_ids,
    #                                dev_ratio=config.dev_ratio,
    #                                test_len=config.test_len,
    #                                seed=config.seed)

    # config.batch_size is useless as we force get_datasets_from_dir return the entire data
    # train_X, train_Y, train_M, valid_X, valid_Y, valid_M, _, _, _ =\
    #     get_datasets_from_dir(sc, config.data_path, config.batch_size,
    #                       train_cells=config.num_cells_train,
    #                       valid_cells=config.num_cells_valid,
    #                       test_cells=config.num_cells_test)[0]
    #
    # dataset = TFDataset.from_ndarrays([train_X, train_M, train_Y], batch_size=batch_size,
    #                                   val_tensors=[valid_X, valid_M, valid_Y],)
    #
    # model = Model(config, dataset.tensors[0], dataset.tensors[1], dataset.tensors[2])

    train_rdd, val_rdd, test_rdd = \
        get_datasets_from_dir_spark(sc, config.data_path, config.batch_size,
                                    train_cells=config.num_cells_train,
                                    valid_cells=config.num_cells_valid,
                                    test_cells=config.num_cells_test)

    dataset = TFDataset.from_rdd(train_rdd,
                                 features=[(tf.float32, [10, 8]), (tf.float32, [77, 8])],
                                 labels=(tf.float32, [8]),
                                 batch_size=config.batch_size,
                                 val_rdd=val_rdd)

    model = Model(config, dataset.tensors[0][0], dataset.tensors[0][1], dataset.tensors[1])

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


