import tensorflow as tf
import os
from utils import get_logger, make_date_dir, find_latest_dir
from data_utils import load_agg_selected_data_mem, batch_loader
import numpy as np
from time import time
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
        _, _, test_x, _, _, test_y, _, _, test_m, test_dt = load_agg_selected_data_mem(data_path=config.data_path, \
            x_len=config.x_len, \
            y_len=config.y_len, \
            foresight=config.foresight, \
            cell_ids=config.test_cell_ids, \
            dev_ratio=config.dev_ratio, \
            test_len=config.test_len, \
            seed=config.seed)
                    
        # please set the latest file path to deploy this test on config file.
        if config.latest_model:
            model_dir = find_latest_dir(os.path.join(config.model, 'model_save/'))
            model_dir = os.path.join(model_dir, 'AR_mem_final/')
        else:
            if not model_dir:
                raise Exception("model_dir or latest_model=True should be defined in config")
            model_dir = config.model_dir
        model = tf.keras.models.load_model(model_dir, compile=False)

        # test function
        @tf.function
        def test_step(train_data, memories, labels):
            preds = model([train_data, memories],training=False)
            t_loss_mse = MSE(labels, preds)
            t_loss_rse = RSE(labels, preds)
            t_loss_smape = SMAPE(labels, preds)
            t_loss_mae = MAE(labels, preds)

            test_loss_mse(t_loss_mse)
            test_accuracy(labels, preds)

        if len(test_y) > 100000:
            # Batch mode
            test_data = list(zip(test_x, test_m, test_y))
            test_batches = batch_loader(test_data, config.batch_size)
            total_pred = np.empty(shape=(0, test_y.shape[1]))

            for batch in test_batches:
                batch_x, batch_m, batch_y = zip(*batch)
                test_step(batch_x, batch_m, batch_y)
                total_pred = np.r_[total_pred, test_loss_mse.result()]
                          
        else:
            # Not batch mode
            test_step(test_x, test_m, test_y)

        result_dir = make_date_dir(os.path.join(config.model, 'results/'))
        np.save(os.path.join(result_dir, 'pred.npy'), total_pred)
        np.save(os.path.join(result_dir, 'test_y.npy'), test_y)
        np.save(os.path.join(result_dir, 'test_dt.npy'), test_dt)
        logger.info("Saving results at {}".format(result_dir))
        logger.info("Testing finished, exit program")
        
    except:
        logger.exception("ERROR")
    
if __name__ == "__main__":
    main()
    
