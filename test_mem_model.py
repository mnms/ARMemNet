import os
import numpy as np
from time import time
import tensorflow as tf
from utils import get_logger, find_latest_dir, make_date_dir
from data_utils import load_agg_selected_data_mem, batch_loader
from AR_mem.config import Config
from AR_mem.model import ARMemNet
from AR_mem.losses import RSE, SMAPE

def main():
    config = Config()

    logger, log_dir = get_logger(os.path.join(config.model, "logs/"))
    logger.info("=======Model Configuration=======")
    logger.info(config.desc)
    logger.info("=================================")

    try:
        # prepare test data
        _, _, test_x, _, _, test_y, _, _, test_m, _ = load_agg_selected_data_mem(data_path=config.data_path, \
            x_len=config.x_len, \
            y_len=config.y_len, \
            foresight=config.foresight, \
            cell_ids=config.train_cell_ids, \
            dev_ratio=config.dev_ratio, \
            test_len=config.test_len, \
            seed=config.seed)

        # metrics
        MSE = tf.keras.losses.MeanSquaredError()
        MAE = tf.keras.losses.MeanAbsoluteError()

        # get (latest) model dir
        if config.latest_model:
            model_dir = find_latest_dir(os.path.join(config.model, 'model_save/'))
        else:
            if not model_dir:
                raise Exception("model_dir or latest_model=True should be defined in config")
            model_dir = config.model_dir
        ckpt_path = os.path.join(model_dir, 'checkpoint')

        # load model
        model = ARMemNet(config)
        model.load_weights(ckpt_path).expect_partial() # exclude optimizer
        logger.info('model weights loaded from {}'.format(ckpt_path))

        # predicts w/ test set
        preds = model([test_x, test_m], training=False)

        # calc metrics
        loss_mse = MSE(test_y, preds)
        loss_rse = RSE(test_y, preds)
        loss_smape = SMAPE(test_y, preds)
        loss_mae = MAE(test_y, preds)

        logger.info("[Inference Result] mse: {:.4f}, rse: {:.4f}, smape: {:.4f}, mae: {:.4f}".format( \
            loss_mse, loss_rse, loss_smape, loss_mae))

        # save results
        result_dir = make_date_dir(os.path.join(config.model, 'results/'))
        np.save(os.path.join(result_dir, 'preds.npy'), preds)
        np.save(os.path.join(result_dir, 'test_y.npy'), test_y)
        logger.info("results are saved at {}".format(result_dir))

    except:
        logger.exception("ERROR")

if __name__ == "__main__":
    main()
