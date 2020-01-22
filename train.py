import os
from time import time

from data_utils import get_logger, make_date_dir, get_datasets_from_dir
from AR_mem.config import Config
from AR_mem.model import Model


def main():
    config = Config()

    logger, log_dir = get_logger(os.path.join(config.model, "logs/"))
    logger.info("=======Model Configuration=======")
    logger.info(config.desc)
    logger.info("=================================")

    try:
        model = Model(config)

        no_improv = 0
        best_loss = 100
        model_dir = make_date_dir(os.path.join(config.model, 'model_save/'))
        logger.info("Start training")

        start_time = time()
        for i in range(config.num_epochs):
            epoch = i + 1

            for train_X, train_Y, train_M, valid_X, valid_Y, valid_M, _, _, _ in \
                    get_datasets_from_dir(config.data_path, config.batch_size,
                                          train_cells=config.num_cells_train,
                                          valid_cells=config.num_cells_valid,
                                          test_cells=config.num_cells_test):

                loss, rse, smape, mae, step = model.train(train_X, train_M, train_Y)

                if step % 100 == 0:
                    logger.info("epoch: %d, step: %d, loss: %.4f, rse: %.4f, smape: %.4f, mae: %.4f" %
                                (epoch, step, loss, rse, smape, mae))

            # dev score for each epoch (no mini batch)
            _, dev_loss, dev_rse, dev_smape, dev_mae = model.eval(valid_X, valid_M, valid_Y)

            if dev_loss < best_loss:
                best_loss = dev_loss
                no_improv = 0
                logger.info("New score! : dev_loss: %.4f, dev_rse: %.4f, dev_smape: %.4f, dev_mae: %.4f" %
                            (dev_loss, dev_rse, dev_smape, dev_mae))
                logger.info("Saving model at {}".format(model_dir))
                model.save_session(os.path.join(model_dir, config.model))
            else:
                no_improv += 1
                if no_improv == config.nepoch_no_improv:
                    logger.info("No improvement for %d epochs" % no_improv)
        #                     break

        elapsed = time() - start_time
        # generating results (no mini batch)
        logger.info("Saving model at {}".format(model_dir))
        logger.info("Elapsed training time {0:0.2f} mins".format(elapsed / 60))
        logger.info("Training finished, exit program")

    except:
        logger.exception("ERROR")


if __name__ == "__main__":
    main()

