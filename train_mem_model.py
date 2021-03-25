import os
import numpy as np
from time import time
import tensorflow as tf
from utils import get_logger, make_date_dir
from data_utils import load_agg_selected_data_mem, batch_loader
from AR_mem.config import Config
from AR_mem.model import Model
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
        train_x, dev_x, _, train_y, dev_y, _, train_m, dev_m, _, _ = load_agg_selected_data_mem(data_path=config.data_path, \
            x_len=config.x_len, \
            y_len=config.y_len, \
            foresight=config.foresight, \
            cell_ids=config.train_cell_ids, \
            dev_ratio=config.dev_ratio, \
            test_len=config.test_len, \
            seed=config.seed)
        train_data = list(zip(train_x, train_m, train_y))

        # prepare training
        logger.info("prepare training...")
        no_improv = 0
        best_loss = 100
        model = Model(config)
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)
        model_dir = make_date_dir(os.path.join(config.model, 'model_save/'))

        MSE = tf.keras.losses.MeanSquaredError()
        MAE = tf.keras.losses.MeanAbsoluteError()

        # training
        logger.info("start training...")
        start_time = time()

        for i in range(config.num_epochs):
            train_batches = batch_loader(train_data, config.batch_size)
            cur_epoch = i + 1

            for step, batch in enumerate(train_batches):
                batch_x, batch_m, batch_y = zip(*batch)

                with tf.GradientTape() as tape:
                    preds = model([batch_x, batch_m], training=True)
                    loss_mse = MSE(batch_y, preds)
                    loss_rse = RSE(batch_y, preds)
                    loss_smape = SMAPE(batch_y, preds)
                    loss_mae = MAE(batch_y, preds)

                    grads = tape.gradient(loss_mse, model.trainable_variables)
                    clipped_grads = tf.clip_by_global_norm(grads, config.clip)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                if step % 50 == 0:
                    logger.info("[T] epoch: {}, step: {}, loss: {:.4f}, rse: {:.4f}, smape: {:.4f}, mae: {:.4f}".format( \
                        cur_epoch, step, loss_mse, loss_rse, loss_smape, loss_mae))

            # evaluate per epoch
            preds_dev = model([dev_x, dev_m], training=False)
            dev_mse = MSE(dev_y, preds_dev)
            dev_rse = RSE(dev_y, preds_dev)
            dev_smape = SMAPE(dev_y, preds_dev)
            dev_mae = MAE(dev_y, preds_dev)

            if dev_mse < best_loss:
                best_loss = dev_mse
                no_improv = 0
                logger.info("New score!: dev_loss: {:.8f}, dev_rse: {:.4f}, dev_smape: {:.4f}, dev_mae: {:.4f}".format( \
                    dev_mse, dev_rse, dev_smape, dev_mae))
                logger.info("Saving model at {}".format(model_dir))
                # tf.keras.models.save_model(model, model_dir, save_format="tf")
                model.save(model_dir)
            else:
                no_improv = no_improv + 1
                if no_improv >= config.nepoch_no_improv:
                    logger.info("No improvement for {} epochs".format(no_improv))
                    break

        #
        elapsed = time() - start_time
        # generating results (no mini batch)
        logger.info("Saving model at {}".format(model_dir))
        logger.info("Elapsed training time {0:0.2f} mins".format(elapsed/60))
        logger.info("Training finished, exit program")

    except:
        logger.exception("ERROR")

if __name__ == "__main__":
    main()
