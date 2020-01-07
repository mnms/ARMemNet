import logging

# Preprocessing Config
class PreprocessConfig:
    def __init__(self):
        # data path
        self.RAW_DATA_DIR = '../data-290k/byCell'       # should contains {CELL_NUM}.csv files
        self.RESULTS_DIR  = '../data-290k-preprocessed'  # will save {CELL_NUM}.npy files

        # preprocess properties
        self.X_SIZE = 10        # # of items per X
        self.X_DAYS = 1         # # of days per X
        self.M_SIZE = 11        # # of items per M
        self.M_DAYS = 7         # # of days per M
        self.M_GAPS = 12*24     # # of rows per day - using 60min / 5min/row x 24hours
        self.Y_SIZE = 1         # # of items per Y

        self.TOTAL_DAYS    = 14 # total days for dataset - 2 weeks
        self.ROWS_PER_CELL = (self.TOTAL_DAYS-self.X_DAYS-(self.M_DAYS-1))*self.M_GAPS-(self.M_SIZE-1) # total rows per cell

        # dataset properties
        self.IDX_COL    = 'CELL_NUM'        # column name for index
        self.FEAT_COLS  = ['RSRP', 'RSRQ', 'DL_PRB_USAGE_RATE', 'SINR',
                            'UE_TX_POWER', 'PHR', 'UE_CONN_TOT_CNT', 'CQI']
        self.N_FEAT     = len(self.FEAT_COLS)    # features per item

        # scaler setting
        self.SCALE_RANGE    = (-1., 1.)
        self.SCALER_DUMP    = '../data-290k/scaler.pkl'

        # error log level
        self.LOG_LEVEL  = logging.INFO
