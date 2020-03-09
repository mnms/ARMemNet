import os
from utils import get_logger, make_date_dir
from data_utils import load_agg_selected_data_mem, batch_loader, BatchDataset
import numpy as np
from time import time
from AR_mem.config import Config
from AR_mem.model import Model

import torch
import torch.utils.data as tud


def main():
    config = Config()
        
    logger, log_dir = get_logger(os.path.join(config.model, "logs/"))
    logger.info("=======Model Configuration=======")
    logger.info(config.desc)
    logger.info("=================================")
    
    try:       
        train_x, dev_x, test_x, train_y, dev_y, test_y, train_m, dev_m, test_m, test_dt = load_agg_selected_data_mem(data_path=config.data_path, \
            x_len=config.x_len, \
            y_len=config.y_len, \
            foresight=config.foresight, \
            cell_ids=config.train_cell_ids, \
            dev_ratio=config.dev_ratio, \
            test_len=config.test_len, \
            seed=config.seed)
                    
        model = Model(config)
        if config.allow_gpu:
            model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        # get train data
        TrainDataSet = BatchDataset(train_x, train_m, train_y)
        TrainSampler = tud.RandomSampler(TrainDataSet)
        TrainDataLoader = tud.DataLoader(TrainDataSet,
                                         batch_size=config.batch_size,
                                         sampler=TrainSampler,
                                         num_workers=2)

        # get valid Data
        dev_x, dev_m, dev_y = torch.Tensor(dev_x), torch.Tensor(dev_m), torch.Tensor(dev_y)

        if config.allow_gpu:
            dev_x, dev_m, dev_y = dev_x.cuda(), dev_m.cuda(), dev_y.cuda()

        step = 0
        no_improv = 0 
        best_loss = 100
        model_dir = make_date_dir(os.path.join(config.model, 'model_save/'))
        logger.info("Start training")
        
        start_time = time()
        for i in range(config.num_epochs):
            epoch = i+1

            # train
            model.train()
            for batch_x, batch_m, batch_y in TrainDataLoader:
                step = step + 1

                if config.allow_gpu:
                    batch_x, batch_m, batch_y = batch_x.cuda(), batch_m.cuda(), batch_y.cuda()

                optimizer.zero_grad()
                prediction, loss, rse, smape, mae = model(batch_x, batch_m, batch_y)

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
                optimizer.step()

                if step % 100 == 0:
                    logger.info("epoch: %d, step: %d, loss: %.4f, rse: %.4f, smape: %.4f, mae: %.4f" %
                                (epoch, step, loss, rse, smape, mae))

            # dev score for each epoch (no mini batch)
            with torch.no_grad():
                model.eval()
                prediction, dev_loss, dev_rse, dev_smape, dev_mae = model(dev_x, dev_m, dev_y)

            if dev_loss < best_loss:
                best_loss = dev_loss
                no_improv = 0
                # logger.info("New score! : dev_loss: %.4f, dev_rse: %.4f, dev_smape: %.4f, dev_mae: %.4f" %
                #             (dev_loss, dev_rse, dev_smape, dev_mae))
                # logger.info("Saving model at {}".format(model_dir))
                torch.save(model, model_dir + "/" + config.model + ".pth")
            else:
                no_improv += 1
                if no_improv == config.nepoch_no_improv:
                    logger.info("No improvement for %d epochs" % no_improv)
                    break

        elapsed = time() - start_time
        # generating results (no mini batch)
        logger.info("Saving model at {}".format(model_dir))
        logger.info("Elapsed training time {0:0.2f} mins".format(elapsed / 60))
        logger.info("Training finished, exit program")

    except:
        logger.exception("ERROR")
    
if __name__ == "__main__":
    main()
    