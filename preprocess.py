import os
import gc
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import PreprocessConfig
from data_utils import get_logger, generate_xym


if __name__ == '__main__':
    # declare preprocessing config
    PRE_CONF = PreprocessConfig()

    # declare logger
    logger, _ = get_logger(log_level=PRE_CONF.LOG_LEVEL)

    # read raw files
    raw_files = os.listdir(PRE_CONF.RAW_DATA_DIR)
    logger.info('Start preprocessing {} files'.format(len(raw_files)))

    # load scaler
    with open(PRE_CONF.SCALER_DUMP, 'rb') as scaler_dump:
        scaler = pickle.load(scaler_dump)

    # read raw file and normalize, then generate x, y, m to save
    for raw_file in tqdm(raw_files):
        # get CELL_NUM from filename
        cell_num = raw_file.split('.')[0]
        target_file = '{}/{}.npz'.format(PRE_CONF.RESULTS_DIR, cell_num)

        # generate result dirs if not exist
        if not os.path.isdir(PRE_CONF.RESULTS_DIR):
            os.makedirs(PRE_CONF.RESULTS_DIR)

        # generate DF from raw csv
        tmp_df = pd.read_csv(os.path.join(PRE_CONF.RAW_DATA_DIR, raw_file), header=0)
        tmp_df['CELL_NUM'] = int(cell_num)
        tmp_df = tmp_df.rename(columns={'evt_dtm': 'EVT_DTM', 'rsrp': 'RSRP', 'rsrq': 'RSRQ',
                                        'dl_prb_usage_rate': 'DL_PRB_USAGE_RATE', 'sinr': 'SINR',
                                        'ue_tx_power': 'UE_TX_POWER', 'phr': 'PHR',
                                        'ue_conn_tot_cnt': 'UE_CONN_TOT_CNT', 'cqi': 'CQI'})

        # Normalzing
        tmp_df[PRE_CONF.FEAT_COLS] = scaler.transform(tmp_df[PRE_CONF.FEAT_COLS])

        # Generate X, Y, M
        tmp_X, tmp_Y, tmp_M = generate_xym(tmp_df[PRE_CONF.FEAT_COLS].to_numpy(), PRE_CONF.N_FEAT, PRE_CONF.X_SIZE, PRE_CONF.Y_SIZE, PRE_CONF.M_SIZE, PRE_CONF.M_DAYS, PRE_CONF.M_GAPS)

        # Save X, Y, M
        # np.save('{}/X.npy'.format(PRE_CONF.RESULTS_DIR), tmp_X)
        # np.save('{}/Y.npy'.format(PRE_CONF.RESULTS_DIR), tmp_Y)
        # np.save('{}/M.npy'.format(PRE_CONF.RESULTS_DIR), tmp_M)
        np.savez_compressed(target_file, X=tmp_X, Y=tmp_Y, M=tmp_M)

        # free used vars
        del tmp_X, tmp_Y, tmp_M
        gc.collect()

    logger.info('Complete preprocessing {} files'.format(len(raw_files)))
