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

    test_x = np.random.rand(config.num_data, 10, 8).astype('float32')
    test_m = np.random.rand(config.num_data, 77, 8).astype('float32')
    test_y = np.random.rand(config.num_data, 8).astype('float32')

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    logger.info("Eager execution    : {}".format(tf.executing_eagerly()))

    checkpoint_status = None
    checkpoint_dir = make_date_dir(os.path.join(config.model, 'model_save/'))
    checkpoint_prefix = os.path.join(checkpoint_dir, config.model)
    print("Current Checkpoint prefix : ", checkpoint_prefix)
    with dist_strategy.scope():
        dist_dataset = ARmemNet_Dataset(config, test_x, test_m, test_y, False, True, seed=config.seed)
        options.experimental_optimization.autotune = True
        test_dataloader  = dist_dataset.test_dataloader().with_options(options)

        test_dist_dataloader = dist_strategy.experimental_distribute_dataset(test_dataloader)
        # Model
        model = ARMemNet(config)

        # Define Optimizer and loss functions
        optimizer = tf.keras.optimizers.Adam(learning_rate=(config.lr))
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        latest_checkpoint_dir = read_latest_dir_from_cache(config.latest_model_dir_cache)
        infer_func = None
        if latest_checkpoint_dir is not None:
            print("[INFO] Lastest checkpoint is in the directory named %s"%(latest_checkpoint_dir))
            #status = checkpoint.restore(tf.train.latest_checkpoint(latest_checkpoint_dir)).assert_existing_objects_matched()
            infer_func = tf.saved_model.load(latest_checkpoint_dir)
        else: 
            print('[WARN] Could not find the cache file that it has latest checkpoint_dir')
            print('[WARN] Training will be start from the scratch!!')
                
    # test function
    @tf.function()
    def test_step(inputs):
        features, memories, labels = inputs
        logger.info("\n\nTest features.shape : {}\n\n".format(features.shape))
        logger.info("Test batch size : {}".format(config.test_batch_size))
        preds = infer_func([features, memories])
        return preds

    # Distributed train function
    @tf.function()
    def distributed_test_step(dist_inputs):
        return dist_strategy.run(test_step, args=(dist_inputs,))
    
    logger.info("Start testing")
    tot_start_time = time.time()
    test_num_batches = 0
    avg_step_elapsed = 0

    # Test phase
    for data in test_dist_dataloader:
        step_start_time = time.time()
        preds= distributed_test_step(data)
        step_elapsed = time.time() - step_start_time
        test_num_batches += 1
        avg_step_elapsed += step_elapsed

    tot_elapsed = time.time() - tot_start_time
    avg_step_elapsed /= test_num_batches
    avg_step_throughput = float(tf.shape(preds.values)[1]) / avg_step_elapsed
    logger.info('===================================================')
    logger.info("[{}] # of GPUs beeing used     :  {}\n".format(rank, tot_elapsed))
    logger.info("[{}] Total elpased time        :  {}\n".format(rank, tot_elapsed))
    logger.info("[{}] perStep elpased time(Avg) :  {}\n".format(rank, avg_step_elapsed))
    logger.info("[{}] Inference throughput(Avg) :  {}\n".format(rank, avg_step_throughput))
    logger.info('===================================================\n\n')
    logger.info("Testing finished, exit program")

if __name__ == "__main__":
    main()
