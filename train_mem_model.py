import os, sys, multiprocessing
import json
import time

from utils import get_logger, get_mpi_info,  make_date_dir, write_latest_dir_to_cache, read_latest_dir_from_cache
from data_utils import load_agg_selected_data_mem, ARmemNet_Dataset, batch_loader
from dist_utils import TFconfig
from AR_mem.config import Config
from AR_mem.model import ARMemNet
from AR_mem.losses import RSE, SMAPE
from tensorflow.keras.losses import MSE, MAE

os.environ['CUDA_AUTO_BOOST'] = '1' 
##Share thread gpu and cpu('global') GPU use dedicated threads('gpu_private') ALL GPUs share a dedicated thread pool('gpu_shared')
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] ='32'  
os.environ['TF_ENABLE_XLA'] = '1'
os.environ.pop('TF_CONFIG', None)
import tensorflow as tf
import numpy as np

def main():
    # App initialization
    config = Config()
    size, rank, host, univ_sz = get_mpi_info()
    local_rank = rank % config.num_gpus_per_node
    print("\n\n<<<<<<<<<<<    np : ", size, "rank: ", rank ,"    >>>>>>>>>>>>>\n\n")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("\n\n[%d] pysical_gpu : "%rank, gpus)

    logger, log_dir = get_logger(os.path.join(config.model, "logs/"))
    logger.info("=======================    Environment Configuration    =======================")
    logger.info("TensorFlow version : {}".format(tf.__version__))
    logger.info("GPU List           :")
    for idx in range(len(gpus)):
        logger.info("  GPU[%d]   =  "%(idx) + '{}'.format(gpus[idx]))
    logger.info("===============================================================================\n\n")
        

    logger.info("=======================        Model Configuration      =======================")
    logger.info(config.desc)
    logger.info("===============================================================================\n\n")

    ## Physical GPUs configuration
    ## Pin GPU to be used to process local rank (one GPU per Process)
    #print("\n\n[%d] set_visible_gpu to >>>> (%d) "%(rank, local_rank))
    #tf.config.experimental.set_visible_devices(gpus[local_rank], 'GPU')
    #print("[%d] Done! set_visible_gpu to >>>> (%d) "%(rank, local_rank))
    vis_gpus = tf.config.experimental.get_visible_devices('GPU')
    print("[%d] visible_gpu(%d) : "%(rank, local_rank), vis_gpus)
    print("\n\n")
    if rank==0:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Distributed Strategy Configuraiton
    dist_strategy = None
    # TF_CONFIG Generator and setting
    tf_config = TFconfig(rank)
    tf_config.generate_config(config, config.ip_list, config.base_port)
    tf_config.reset_config()
    tf_config.print_config()

    print("\n\n[%d] Deive Configuration Done\n\n"%rank)

    # Distributed strategy Configuration
    comm_opt = tf.distribute.experimental.CommunicationOptions( \
                        implementation = tf.distribute.experimental.CommunicationImplementation.NCCL \
                    )
    dist_strategy = tf.distribute.MultiWorkerMirroredStrategy( \
                        communication_options=comm_opt \
                    )
    print("\n\n[%d] Distributed strategy Configuration Done\n\n"%rank)

    train_x = np.random.rand(config.num_data, 10, 8).astype('float32')
    train_m = np.random.rand(config.num_data, 77, 8).astype('float32')
    train_y = np.random.rand(config.num_data, 8).astype('float32')

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    logger.info("Eager execution    : {}".format(tf.executing_eagerly()))

    checkpoint_dir = make_date_dir(os.path.join(config.model, 'model_save/'))
    checkpoint_prefix = os.path.join(checkpoint_dir, config.model)
    print("Current Checkpoint prefix : ", checkpoint_prefix)

    with dist_strategy.scope():
        dist_dataset = ARmemNet_Dataset(config, train_x, train_m, train_y, False, True, seed=config.seed)
        options.experimental_optimization.autotune = True
        train_dataloader = dist_dataset.train_dataloader().with_options(options)
        test_dataloader  = dist_dataset.test_dataloader().with_options(options)

        train_dist_dataloader = dist_strategy.experimental_distribute_dataset(train_dataloader)
        test_dist_dataloader = dist_strategy.experimental_distribute_dataset(test_dataloader)
        # Model
        model = ARMemNet(config)


        # Define Optimizer and loss functions
        optimizer = tf.keras.optimizers.Adam(learning_rate=(config.lr))

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        if config.latest_model is True:
            latest_checkpoint_dir = read_latest_dir_from_cache(config.latest_model_dir_cache)
            if latest_checkpoint_dir is not None:
                print("[INFO] Lastest checkpoint is in the directory named %s"%(latest_checkpoint_dir))
                status = checkpoint.restore(tf.train.latest_checkpoint(latest_checkpoint_dir)).assert_existing_objects_matched()
            else: 
                print('[WARN] Could not find the cache file that it has latest checkpoint_dir')
                print('[WARN] Training will be start from the scratch!!')
                

        # Loss functions
        test_loss = tf.keras.metrics.Mean()

        # accuracy metrics
        train_accuracy = tf.keras.metrics.CategoricalCrossentropy()
        test_accuracy  = tf.keras.metrics.CategoricalCrossentropy()

        # Compute loss function
        @tf.function()
        def compute_losses_accuracy(labels, predictions, global_batch_size):
            loss_mse = tf.nn.compute_average_loss( MSE(labels, predictions), None, global_batch_size )
            loss_rse = tf.reduce_sum( RSE(labels, predictions) / global_batch_size )
            loss_mae = tf.nn.compute_average_loss( MAE(labels, predictions), None, global_batch_size )
            loss_smape = tf.reduce_sum( SMAPE(labels, predictions) / global_batch_size )
            return loss_mse, loss_rse, loss_mae, loss_smape

    # train function
    @tf.function()
    def train_step(inputs):
        features, memories, labels = inputs
        logger.info("\n\nTrain features.shape : {}".format(features.shape))
        logger.info("Train batch size : {}".format(config.train_batch_size))
        with tf.GradientTape() as tape:
            preds = model([features, memories], training=True)
            loss_mse, loss_rse, loss_mae, loss_smape = compute_losses_accuracy(labels, preds, config.train_batch_size)

        grads = tape.gradient(loss_mse, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_accuracy.update_state(labels, preds)
        return loss_mse

    # test function
    @tf.function()
    def test_step(inputs):
        features, memories, labels = inputs
        global_batch_size = config.test_batch_size * dist_strategy.num_replicas_in_sync
        logger.info("\n\nTest features.shape : {}\n\n".format(features.shape))
        logger.info("Test batch size : {}".format(config.test_batch_size))
        preds = model([features, memories], training=False)
        loss_mse, a, b, c = compute_losses_accuracy(labels, preds, config.test_batch_size)
        test_loss.update_state(loss_mse)
        test_accuracy.update_state(labels, preds)
        return loss_mse

    # Distributed train function
    @tf.function()
    def distributed_train_step(dist_inputs):
        per_replica_losses = dist_strategy.run(train_step, args=(dist_inputs,))
        return dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    # Distributed train function
    @tf.function()
    def distributed_test_step(dist_inputs):
        dist_strategy.run(test_step, args=(dist_inputs,))
    
    no_improv = 0
    best_loss = 100
    logger.info("Start training")
    tot_start_time = time.time()
    for epoch_idx in range(1, config.num_epochs):
        train_total_loss = 0.0
        train_num_batches = 0
        test_total_loss = 0.0
        test_num_batches = 0

        epoch_start_time = time.time()
        # Training phase
        for data in train_dist_dataloader:
            step_start_time = time.time()
            train_total_loss += distributed_train_step(data)
            train_num_batches += 1
            step_elapsed = time.time() - step_start_time
            print("\n[{}] Train step #{} :  {}\n".format(rank, train_num_batches, step_elapsed))
        train_loss = train_total_loss / train_num_batches

        # Test phase
        for data in test_dist_dataloader:
            step_start_time = time.time()
            distributed_test_step(data)
            step_elapsed = time.time() - step_start_time
            print("\n[{}] Test step #{} :  {}\n".format(rank, train_num_batches, step_elapsed))

        epoch_elapsed = time.time() - epoch_start_time
        if (rank == 0) and (best_loss > test_loss.result()) :
            logger.info("New score! : dev_loss: {:2.8f}, dev_rse: {:2.8f} >> {}".format(best_loss, test_loss.result(), (best_loss -test_loss.result())))
            save_start_time = time.time()
            best_loss = test_loss.result()
            saved_path = checkpoint.save(checkpoint_prefix)
            save_elapsed = time.time() - save_start_time
            logger.info("Saved checkpoint for epoch: {}  >>>  {}".format(epoch_idx, saved_path))
            write_latest_dir_to_cache(config.latest_model_dir_cache, checkpoint_dir)
            logger.info("It took {} secs for saving !!!".format(save_elapsed))
            no_improv = 0
        else:
            if no_improv >= config.nepoch_no_improv:
                break
            no_improv += 1

        logger.info('===================================================')
        logger.info('Epoch {}, {:3.4f} sec.'.format(epoch_idx, epoch_elapsed))
        logger.info('Loss: {}, '.format(train_loss))
        logger.info('Accuracy: {}, '.format(train_accuracy.result() * 100))
        logger.info('Test Loss: {}, '.format(test_loss.result()))
        logger.info('Test Accuracy: {}'.format(test_accuracy.result() * 100))
        logger.info('===================================================\n\n')
    

    tot_elapsed = time.time() - tot_start_time
    if (rank==0):
        tf.saved_model.save(model, checkpoint_dir)
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()

    # generating results 
    logger.info("Elapsed training time {:3.4f} mins".format(tot_elapsed/60))
    logger.info("Training finished, exit program")

if __name__ == "__main__":
    main()
