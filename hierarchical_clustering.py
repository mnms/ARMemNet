import os
from data_utils import load_agg_data, load_agg_data_all, batch_loader
from utils import get_logger, find_latest_dir
import numpy as np
from Seq_model.config import Config
from Seq_model.model import Model
import scipy.cluster.hierarchy as shc
from matplotlib import pyplot as plt


def main():
    config = Config()
    logger, log_dir = get_logger(os.path.join(config.model, "logs/"))
    logger.info("=======Model Configuration=======")
    logger.info(config.desc)
    logger.info("=================================")

    try:
        full_data = load_agg_data_all(data_path=config.data_path, ncells=config.ncells, test_len=config.test_len)
        
        model = Model(config)
        if config.latest_model:
            model_dir = find_latest_dir(os.path.join(config.model, 'model_save/'))
        else:
            if not model_dir:
                raise Exception("model_dir or latest_model=True should be defined in config")
            model_dir = config.model_dir

        # Launch the graph
        model.restore_session(model_dir)   
        total_states = model.extract(full_data)
        total_states = np.squeeze(total_states, axis=0)
        
        result_dir = os.path.join(config.model, 'cell_vectors/')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        np.save(os.path.join(result_dir, 'test_total_states.npy'), total_states)
        logger.info("Saving vectors at {}".format(result_dir))
        logger.info("Finish extracting vectors")
        logger.info("Start hierarchical clustering")

        
        # total_states = np.load(os.path.join(result_dir, 'test_total_states.npy'))
        states = total_states

        plt.figure(figsize=(10, 7))  
        plt.title("Dendrograms")  
        dend = shc.dendrogram(shc.linkage(states, method='ward'))

        plt_dir = 'clustering_plots/'
        if not os.path.exists(plt_dir):
            os.mkdir(plt_dir)
        plt.savefig(os.path.join(plt_dir,'hierarchical_clustering.png'))

    except:
        logger.exception("ERROR")


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("-g", "--gpu", dest="gpu_index", help="gpu index", default="0", type=str)
    # args = parser.parse_args()
    # print("CUDA_VISIBLE_DEVICES: {}".format(args.gpu_index))
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index

    main()
    
    



