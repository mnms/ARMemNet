import os
from utils import get_logger, make_date_dir, find_latest_dir
from data_utils import load_agg_selected_data_mem, batch_loader
import numpy as np
from time import time
from AR_mem.config import Config
from AR_mem.model import Model
import torch


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
                    
        model = Model(config)

        test_x, test_m, test_y = torch.Tensor(test_x), torch.Tensor(test_m), torch.Tensor(test_y)

        if config.allow_gpu:
            model = model.cuda()
            test_x, test_m, test_y = test_x.cuda(), test_m.cuda(), test_y.cuda()

        if config.latest_model:
            model_dir = find_latest_dir(os.path.join(config.model, 'model_save/'))
        else:
            if not model_dir:
                raise Exception("model_dir or latest_model=True should be defined in config")
            model_dir = config.model_dir
        model = torch.load(model_dir + "/" + config.model + ".pth")

        with torch.no_grad():
            model.eval()

            if len(test_y) > 100000:
                # Batch mode
                test_data = list(zip(test_x, test_m, test_y))
                test_batches = batch_loader(test_data, config.batch_size)
                total_pred = np.empty(shape=(0, test_y.shape[1]))

                for batch in test_batches:
                    batch_x, batch_m, batch_y = zip(*batch)
                    pred, _, _, _, _ = model(batch_x, batch_m, batch_y)
                    total_pred = np.r_[total_pred, pred]
            else:
                # Not batch mode
                total_pred, test_loss, test_rse, test_smape, test_mae = model(test_x, test_m, test_y)

        logger.info("MAE: {}".format(test_mae))
        result_dir = make_date_dir(os.path.join(config.model, 'results/'))
        np.save(os.path.join(result_dir, 'pred.npy'), total_pred.cpu())
        np.save(os.path.join(result_dir, 'test_y.npy'), test_y)
        np.save(os.path.join(result_dir, 'test_dt.npy'), test_dt)
        logger.info("Saving results at {}".format(result_dir))
        logger.info("Testing finished, exit program")

    except:
        logger.exception("ERROR")
    
if __name__ == "__main__":
    main()
