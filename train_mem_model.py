import tensorflow as tf
import horovod.tensorflow as hvd

import os
from utils import get_logger, make_date_dir
from data_utils import load_agg_selected_data_mem, batch_loader
import numpy as np
from time import time
from AR_mem.config import Config
from AR_mem.model import Model
from AR_mem.losses import RSE, SMAPE

def main():
    #Initialize Horovod
    hvd.init()

    #Pin GPU to be used to process local rank (one GPU per Process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')



    config = Config()

        
    logger, log_dir = get_logger(os.path.join(config.model, "logs/"))
    logger.info("=======Model Configuration=======")
    logger.info(config.desc)
    logger.info("=================================")
    
    try:       
        # Data preparation
        # [TODO] Need to apply horovod distributed environment for data loader !!!!!!!
        # For example, 
        # Call the get_dataset function you created, this time with the Horovod rank and size
        # (x_train, y_train), (x_test, y_test) = get_dataset(num_classes, hvd.rank(), hvd.size())
        train_x, dev_x, test_x, train_y, dev_y, test_y, train_m, dev_m, test_m, test_dt = load_agg_selected_data_mem(data_path=config.data_path, \
            x_len=config.x_len, \
            y_len=config.y_len, \
            foresight=config.foresight, \
            cell_ids=config.train_cell_ids, \
            dev_ratio=config.dev_ratio, \
            test_len=config.test_len, \
            seed=config.seed)
        train_data = list(zip(train_x, train_m, train_y))

        # Model
        model_dir = make_date_dir(os.path.join(config.model, 'model_save/'))
        model_file = os.path.join(model_dir, config.model)
        model=None
        if os.path.exists(model_file):
            model = tf.keras.models.load_model(model_file)
        else:
            model = Model(config)



        # Define Optimizer and loss functions
        optimizer = tf.keras.optimizers.Adam(learning_rate=(config.lr*hvd.size()))
        # checkpointer setting
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        status = checkpoint.restore(tf.train.latest_checkpoint(model_file))

        MSE = tf.keras.losses.MeanSquaredError()
        MAE = tf.keras.losses.MeanAbsoluteError()
        train_loss_mse   = tf.keras.metrics.Mean(name='train_loss_mse')
        train_loss_rse   = tf.keras.metrics.Mean(name='train_loss_rse')
        train_loss_smape = tf.keras.metrics.Mean(name='train_loss_smap')
        train_loss_mae   = tf.keras.metrics.Mean(name='train_loss_mae')
        train_accuracy   = tf.keras.metrics.Accuracy(name='train_accuracy')
        test_loss_mse   = tf.keras.metrics.Mean(name='test_loss_mse')
        test_loss_rse   = tf.keras.metrics.Mean(name='test_loss_rse')
        test_loss_smape = tf.keras.metrics.Mean(name='test_loss_smap')
        test_loss_mae   = tf.keras.metrics.Mean(name='test_loss_mae')
        test_accuracy   = tf.keras.metrics.Accuracy(name='test_accuracy')

        # train function
        @tf.function
        def train_step(train_data, memories, labels, first_batch):
            with tf.GradientTape as tape:
                preds = model(train_data, memories, training=True)
                loss_mse = MSE(labels, preds)
                loss_rse = RSE(labels, preds)
                loss_smape = SMAPE(labels, preds)
                loss_mae = MAE(labels, preds)
            # Horovod: add Horovod Distributed GradientTape
            tape = hvd.DistributedGradientTape(tape)

            grads = tape.gradient(loss_mse, model.trainable_variables)
            clipped_grads = tf.clip_by_global_norm(grads, config.clip)
            optimizer.apply_gradient(zip(grads, model.trainable_variables))

            # Horovod: broadcast initial variable states from rank 0 to all other processes.
            # This is necessary to ensure consistent initialization of all workers when
            # training is started with random weights or restored from a checkpoint.
            #
            # Note: broadcast should be done after the first gradient step to ensure optimizer
            # initialization.
            if first_batch:
                hvd.broadcast_variables(model.variables, root_rank=0)
                hvd.broadcast_variables(optimizer.variables(), root_rank=0)

            train_loss_mse(loss_mse)
            train_loss_rse(loss_rse)
            train_loss_smape(loss_smape)
            train_loss_mae(loss_mae)
            train_accuracy(labels, preds)

        # test function
        @tf.function
        def test_step(train_data, memories, labels):
            preds = model(train_data, memories,training=False)
            t_loss_mse = MSE(labels, preds)
            t_loss_rse = RSE(labels, preds)
            t_loss_smape = SMAPE(labels, preds)
            t_loss_mae = MAE(labels, preds)

            test_loss_mse(t_loss_mse)
            test_loss_rse(t_loss_rse)
            test_loss_smape(t_loss_smape)
            test_loss_mae(t_loss_mae)
            test_accuracy(labels, preds)
                

        no_improv = 0 
        best_loss = 100
        logger.info("Start training")
        start_time = time()
        for i in range(config.num_epochs):
            train_batches = batch_loader(train_data, config.batch_size)
            test_batches = batch_loader(test_data, config.batch_size)
            epoch = i+1

            test_loss_mse.reset_states()
            test_loss_rse.reset_states()
            test_loss_smape.reset_states()
            test_loss_mae.reset_states()
            test_accuracy.reset_states()
            train_loss_mse.reset_states()
            train_loss_rse.reset_states()
            train_loss_smape.reset_states()
            train_loss_mae.reset_states()
            train_accuracy.reset_states()

            # Horovod: adjust number of steps based on number of GPUs.
            for batch_idx, batch_train in enumerate(train_batches.take(1000 // hvd.size())):
                batch_x, batch_m, batch_y = zip(*batch_train)
                train_step(batch_x, batch_m, batch_y, batch_idx == 0)

                if batch_idx % 100 == 0:
                    logger.info("epoch: %d, step: %d, loss: %.4f, rse: %.4f, smape: %.4f, mae: %.4f" %
                                (epoch+1, batch_idx, train_loss_mse.result(), train_loss_rse.result() \
                                        , train_loss_smape.result(), train_loss_mae.result()))
            
            for batch_idx, batch_test in enumerate(test_batches.take(1000// hvd.size())):
                batch_x, batch_m, batch_y = zip(*batch_test)
                test_step(batch_x, batch_m, batch_y)

            if test_loss_mse.result() < best_loss:
                best_loss = test_loss_mse.result()
                no_improv = 0
                if hvd.local_rank() == 0:
                    logger.info("New score! : dev_loss: %.4f, dev_rse: %.4f, dev_smape: %.4f, dev_mae: %.4f" % 
                                (test_loss_mse.result(), test_loss_rse.result(), test_loss_smape.result(), test_loss_mae.result()))
                if hvd.rank() == 0:
                    logger.info("Saving model at {}".format(model_dir))
                    model.save(model_file + '_epoch%d'%(i))
            else: 
                no_improv += 1
                if no_improv == config.nepoch_no_improv:
                    logger.info("No improvement for %d epochs" % no_improv)
                    break
            logger.info('\n===================================================')
            logger.info('Epoch {}, '.format(epoch+1))
            logger.info('Loss: {}, '.format(train_loss_mse.result()))
            logger.info('Accuracy: {}, '.format(train_accuracy.result() * 100))
            logger.info('Test Loss: {}, '.format(test_loss_mse.result()))
            logger.info('Test Accuracy: {}'.format(test_accuracy.result() * 100))
            logger.info('===================================================\n\n')
        
        elapsed = time() - start_time
        # generating results (no mini batch)
        logger.info("Saving model at {}".format(model_dir))
        logger.info("Elapsed training time {0:0.2f} mins".format(elapsed/60))
        logger.info("Training finished, exit program")
        if hvd.rank() == 0:
            logger.info("Saving final model at {}".format(model_dir))
            model.save(model_file + '_final')
        
    except:
        logger.exception("ERROR")
    
if __name__ == "__main__":
    main()
    
