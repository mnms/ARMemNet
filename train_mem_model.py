import os
import numpy as np
from time import time
import tensorflow as tf
from utils import get_logger, make_date_dir
from data_utils import load_agg_selected_data_mem, batch_loader
from AR_mem.config import Config
from AR_mem.model import ARMemNet
from AR_mem.losses import RSE, SMAPE
from tensorflow.keras.losses import MSE, MAE


def main():
    config = Config()

    logger, log_dir = get_logger(os.path.join(config.model, "logs/"))
    logger.info("=======Model Configuration=======")
    logger.info(config.desc)
    logger.info("=================================")

    try:
        # prepare train, evaluation data
        # train_x, dev_x, _, train_y, dev_y, _, train_m, dev_m, _, _ = load_agg_selected_data_mem(data_path=config.data_path, \
        train_x, dev_x, test_x, train_y, dev_y, test_y, train_m, dev_m, test_m, _ = load_agg_selected_data_mem(
            data_path=config.data_path, \
            x_len=config.x_len, \
            y_len=config.y_len, \
            foresight=config.foresight, \
            cell_ids=config.train_cell_ids, \
            dev_ratio=config.dev_ratio, \
            test_len=config.test_len, \
            seed=config.seed)

        # define model
        model = ARMemNet(config)
        model_dir = make_date_dir(os.path.join(config.model, 'model_save/'))
        ckpt_path = os.path.join(model_dir, 'checkpoint')

        model.compile( \
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.lr), \
            loss=MSE, \
            metrics=[MSE, RSE, SMAPE, MAE] \
        )

        callbacks = [ \
            tf.keras.callbacks.EarlyStopping(patience=config.nepoch_no_improv, monitor='val_loss'), \
            tf.keras.callbacks.TensorBoard(log_dir=make_date_dir(os.path.join(config.model, 'model_log/'))), \
            tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, save_weights_only=True, monitor='val_loss',
                                               mode='max', save_best_only=False)
        ]

        # train model
        model.fit( \
            [train_x, train_m], train_y, \
            batch_size=config.batch_size, \
            epochs=config.num_epochs, \
            callbacks=callbacks, \
            validation_data=([dev_x, dev_m], dev_y) \
        )

    except:
        logger.exception("ERROR")

if __name__ == "__main__":
    main()
