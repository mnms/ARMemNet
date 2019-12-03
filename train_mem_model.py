import os
from utils import get_logger, make_date_dir
from data_utils import load_agg_selected_data_mem_train, batch_loader
import numpy as np
from time import time
from AR_mem.config import Config
from AR_mem.model import Model


def main():
    config = Config()

    logger, log_dir = get_logger(os.path.join(config.model, "logs/"))
    logger.info("=======Model Configuration=======")
    logger.info(config.desc)
    logger.info("=================================")

    try:
        train_x, dev_x, _, train_y, dev_y, _, train_m, dev_m, _, _ = load_agg_selected_data_mem_train(data_path=config.data_path, \
            x_len=config.x_len, \
            y_len=config.y_len, \
            foresight=config.foresight, \
            cell_ids=config.train_cell_ids, \
            dev_ratio=config.dev_ratio, \
            test_len=config.test_len, \
            seed=config.seed)

        model = Model(config)
        train_data = list(zip(train_x, train_m, train_y))

        no_improv = 0
        best_loss = 100
        model_dir = make_date_dir(os.path.join(config.model, 'model_save/'))
        logger.info("Start training")

        start_time = time()
        logger.info("Training started with Batch Size: {}, Total Epochs: {}".format(config.batch_size, config.num_epochs))
        for i in range(config.num_epochs):
            train_batches = batch_loader(train_data, config.batch_size)
            epoch = i+1

            for batch in train_batches:
                batch_x, batch_m, batch_y = zip(*batch)
                loss, rse, smape, mae, step = model.train(batch_x, batch_m, batch_y)

                if epoch % 50 == 0 and step % 10 == 0:
                    logger.info("epoch: %d, step: %d, loss: %.4f, rse: %.4f, smape: %.4f, mae: %.4f" % (epoch, step, loss, rse, smape, mae))

            # dev score for each epoch (no mini batch)
            _, dev_loss, dev_rse, dev_smape, dev_mae = model.eval(dev_x, dev_m, dev_y)

        elapsed = time() - start_time
        # generating results (no mini batch)
        logger.info("Saving model at {}".format(model_dir))
        logger.info("Elapsed training time {0:0.2f} secs".format(elapsed))
        logger.info("Training fisinised with Batch Size: {0:4d}, Total Epochs: {1:4d}, Elapsed Time: {2:0.2f} secs".format(config.batch_size, config.num_epochs, elapsed))
        logger.info("Training finished, exit program")

    except:
        logger.exception("ERROR")

if __name__ == "__main__":
    main()
