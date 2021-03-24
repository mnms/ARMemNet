# AR_mem_config


class Config(object):
    def __init__(self):
        # model params
        self.model = "AR_mem"
        self.nsteps = 10   # equivalent to x_len
        self.msteps = 7
        self.attention_size = 16
        self.l2_lambda = 1e-3
        self.ar_lambda = 0.1    
        self.ar_g = 1           
        
        # data params
        self.data_path = '../data/aggregated_data_5min_scaled.csv'
        self.nfeatures = 8  # number of col_list in "../config_preprocess.py"
        self.x_len = self.nsteps
        self.y_len = 1
        self.foresight = 0 
        self.dev_ratio = 0.1
        self.train_test_ratio = 0.7
        self.test_len = 7
        self.seed = None
        
        # train & test params
        self.train_cell_ids = [11, 16, 18]  # order of cell_id in "../config_preprocess.py"
        self.test_cell_ids = [18]           # order of cell_id in "../config_preprocess.py"
        self.model_dir = None       # Model directory to use in test mode. For example, "model_save/20190405-05"
        self.latest_model = True    # Use lately saved model in test mode. If latest_model=True, model_dir option will be ignored
        self.latest_model_dir_cache = 'AR_mem/model_save/latest.cache'
        self.latest_model_file = None # please set file path before deploying test_mem_model

        self.num_data   = 1000000
        self.train_data_size = int(self.num_data * self.train_test_ratio)
        self.train_batch_size = int(self.train_data_size / 2)
        self.train_prefetch_size = 32
        self.test_data_size = int(self.num_data * (1-self.train_test_ratio))
        self.test_batch_size = int(self.test_data_size / 2)
        self.test_prefetch_size = 16 

        
        # training params
        self.lr = 1e-3
        self.num_epochs = 50
        self.dropout = 0.8     
        self.nepoch_no_improv = 5
        self.clip = 5
        self.allow_gpu = True
        self.desc = self._desc()

        # Distributed paramas
        self.num_node  = 1
        self.ip_list   = ['localhost']
        self.base_port = 3333 
        self.num_gpus_per_node = 4
        self.num_workers = 1

            
    def _desc(self):
        desc = ""
        for mem, val in self.__dict__.items():
            desc += mem + ":" + str(val) + ", "
        return desc


if __name__ == "__main__":
    config = Config()
    print(config.desc)
