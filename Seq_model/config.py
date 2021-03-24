class Config(object):
    def __init__(self):
        # model params
        self.model = "Seq_model"
        self.nsteps = 7       # equivalent to x_len
        self.hidden_size = 8
        self.gru_size = 8
        self.l2_lambda = 1e-3

        # data params
        self.ncells = 20
        self.data_path = '../data/aggregated_data_5min_scaled.csv'
        self.nfeatures = 8 #6
        self.x_len = self.nsteps
        self.y_len = 1
        self.foresight = 0
        self.dev_ratio = 0.1
        self.test_len = 7 #1236
        self.seed = None

        # train & test params
        self.model_dir = None       # Model directory to use in test mode. For example, "model_save/20190405-05"
        self.latest_model = True    # Use lately saved model in test mode. If latest_model=True, model_dir option will be ignored

        # train params
        self.lr = 1e-3
        self.num_epochs = 100
        self.batch_size = 64
        self.dropout = 0.8
        self.nepoch_no_improv = 5
        self.clip = 5
        self.desc = self._desc()
        self.allow_gpu = True    
        
    def _desc(self):
        desc = ""
        for mem, val in self.__dict__.items():
            desc += mem + ":" + str(val) + ", "
        return desc

if __name__ == "__main__":
    config = Config()
    print(config.desc)


