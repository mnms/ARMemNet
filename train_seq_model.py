import os
from utils import get_logger, make_date_dir
from data_utils import load_agg_data, batch_loader
import numpy as np
from time import time
from Seq_model.config import Config
from Seq_model.model import Model

def main():
    config = Config()

    logger, log_dir = get_logger(os.path.join(config.model, "logs/"))
    logger.info("=======Model Configuration=======")
    logger.info(config.desc)
    logger.info("=================================")
    
    try:       
        train_x, dev_x, test_x, train_y, dev_y, test_y, test_dt = load_agg_data(data_path=config.data_path, \
            x_len=config.x_len, \
            y_len=config.y_len, \
            ncells=config.ncells, \
            foresight=config.foresight, \
            dev_ratio=config.dev_ratio,\
            test_len=config.test_len, \
            seed=config.seed)
            
        model = Model(config)
        train_data = list(zip(train_x,train_y))
        no_improv = 0 
        best_loss = 100
        model_dir = make_date_dir(os.path.join(config.model, 'model_save/'))
        result_dir = make_date_dir(os.path.join(config.model, 'results/'))
        logger.info("Start training")
        dev_x = np.asarray(dev_x)
        dev_y = np.asarray(dev_y)
                
        start_time = time()
        for i in range(config.num_epochs):
            train_batches = batch_loader(train_data, config.batch_size)
            epoch = i+1
            
            for batch in train_batches:
                batch_x, batch_y = zip(*batch)
                batch_x = np.asarray(batch_x)
                batch_y = np.asarray(batch_y)
                loss, rse, smape, mae, step = model.train(batch_x, batch_y)

                if step % 100 == 0:
                    logger.info("epoch: %d, step: %d, loss: %.4f, rse: %.4f, smape: %.4f, mae: %.4f" %
                                (epoch, step, loss, rse, smape, mae))
                    
            # dev score for each epoch (no mini batch)
            _, dev_loss, dev_rse, dev_smape, dev_mae = model.eval(dev_x, dev_y)
        
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
                #    break
        # model.save_session(os.path.join(model_dir, config.model))
        elapsed = time() - start_time
        # generating results (no mini batch)
        model.restore_session(model_dir)
        pred, test_loss, test_rse, test_smape, test_mae = model.eval(test_x, test_y)
        logger.info("test_loss: %.4f, test_rse: %.4f, test_smape: %.4f, test_mae: %.4f" % 
                    (test_loss, test_rse, test_smape, test_mae))
        
        # save results
        # np.save(os.path.join(result_dir, 'pred.npy'), pred)
        # np.save(os.path.join(result_dir, 'test_y.npy'), test_y)
        # np.save(os.path.join(result_dir, 'test_dt.npy'), test_dt)
        # logger.info("Saving results at {}".format(result_dir))
        logger.info("Elapsed training time {0:0.2f} mins".format(elapsed/60))
        logger.info("Training finished, exit program")
        
    except:
        logger.exception("ERROR")
    
if __name__ == "__main__":
    main()
    
