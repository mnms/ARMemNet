import gc
import os
import math
import logging
import logging.handlers
import datetime

import numpy as np
from tqdm import tqdm

# generate rolling window with 'width' from 'np_arr'
def rolling_window(np_arr, width=11):
    n = np_arr.shape[0]
    return np.hstack([np_arr[i:1 + n + i - width] for i in range(0, width)])


# wrap 'np_arr' in 'width' using a rolling_window func and then bind rows that are 'gapsize' apart by 'winsize'
def slided_window(np_arr, width=11, winsize=7, gapsize=288):
    total_rows = np_arr.shape[0]
    result_rows = total_rows - (width - 1) - (winsize - 1) * gapsize  # assume that x_days = 1

    rolled_np = rolling_window(np_arr, width)
    slided_np = np.array([rolled_np[i:i + (winsize - 1) * gapsize + 1:gapsize] for i in range(0, result_rows)])

    # free used var
    del rolled_np
    gc.collect()

    return slided_np


# generate x, y, m as ndarray with given params
def generate_xym(np_arr, n_feat=8, x_size=10, y_size=1, m_size=11, m_days=7, m_gaps=288):
    # generate X / Y
    XY_raw = np_arr[m_days * m_gaps:]
    XY = rolling_window(XY_raw, x_size + y_size)
    X = XY[:, :-1 * n_feat]
    Y = XY[:, -1 * n_feat:]

    # generate M
    M_raw = np_arr[:-m_gaps]
    M = slided_window(M_raw, m_size, m_days, m_gaps)

    # free used var
    del np_arr
    del XY_raw
    del XY
    del M_raw
    gc.collect()

    # return as numpy.ndarray
    return X, Y, M


# get dataset from given filenames
def read_npz_files(preprocessed_dir, files_to_read):
    X, Y, M = None, None, None

    for filename in files_to_read:
        read_npz = np.load(os.path.join(preprocessed_dir, filename))

        if X is None:
            X, Y, M = read_npz['X'], read_npz['Y'], read_npz['M']
        else:
            X = np.vstack((X, read_npz['X']))
            Y = np.vstack((Y, read_npz['Y']))
            M = np.vstack((M, read_npz['M']))

    return X.reshape(-1, 10, 8), Y.reshape(-1, 8), M.reshape(-1, 77, 8)


# get dataset from given preprocessed_dir
def get_datasets_from_dir(preprocessed_dir, batch_size, train_cells=1.0, valid_cells=0, test_cells=0):
    # logger
    logger = logging.getLogger()

    # load preprocessed files from dir & get total rows
    preprocessed_files = os.listdir(preprocessed_dir)
    n_preprocessed_files = len(preprocessed_files)

    # split train / valid / test set
    if train_cells <= 1.0:
        n_train_set = round(n_preprocessed_files * train_cells)
    else:
        n_train_set = int(train_cells)

    if valid_cells <= 1.0:
        n_valid_set = round(n_preprocessed_files * valid_cells)
    else:
        n_valid_set = int(valid_cells)

    if test_cells <= 1.0:
        n_test_set = round(n_preprocessed_files * test_cells)
    else:
        n_test_set = int(test_cells)

    # split by index
    idx_cells = np.random.permutation(n_preprocessed_files)
    idx_train = idx_cells[:n_train_set]
    idx_valid = idx_cells[n_train_set:n_train_set + n_valid_set]
    idx_test = idx_cells[n_train_set + n_valid_set:n_train_set + n_valid_set + n_test_set]

    train_files = [preprocessed_files[j] for j in idx_train]
    valid_files = [preprocessed_files[j] for j in idx_valid]
    test_files = [preprocessed_files[j] for j in idx_test]

    assert n_train_set + n_valid_set + n_test_set <= n_preprocessed_files

    # get valid sets & test sets
    valid_X, valid_Y, valid_M = read_npz_files(preprocessed_dir, valid_files)
    test_X, test_Y, test_M = read_npz_files(preprocessed_dir, test_files)

    # define train_set properties
    n_rows_per_file = np.load(os.path.join(preprocessed_dir, train_files[0]))['X'].shape[0]
    n_total_rows = n_train_set * n_rows_per_file

    # log dataset info
    logger.info('')
    logger.info('Dataset Summary')
    logger.info(' - Used {:6d} cells of {:6d} total cells ({:2.2f}%)'.format(n_train_set + n_valid_set + n_test_set,
                                                                             n_preprocessed_files, (
                                                                                         n_train_set + n_valid_set + n_test_set) / n_preprocessed_files * 100))
    logger.info(' - Train Dataset: {:6d} cells ({:02.2f}% of used cells)'.format(n_train_set, n_train_set / (
                n_train_set + n_valid_set + n_test_set) * 100))
    logger.info(' - Valid Dataset: {:6d} cells ({:02.2f}% of used cells)'.format(n_valid_set, n_valid_set / (
                n_train_set + n_valid_set + n_test_set) * 100))
    logger.info(' - Test Dataset : {:6d} cells ({:02.2f}% of used cells)'.format(n_test_set, n_test_set / (
                n_train_set + n_valid_set + n_test_set) * 100))
    logger.info('')
    logger.info('Trainset Summary')
    logger.info(' - Row / Cell: {:9d} rows / cell'.format(n_rows_per_file))
    logger.info(' - Train Cell: {:9d} cells'.format(n_train_set))
    logger.info(' - Total Rows: {:9d} rows'.format(n_total_rows))
    logger.info(' - Batch Size: {:9d} rows / batch'.format(batch_size))
    logger.info(' - Batch Step: {:9d} batches / epoch'.format(math.ceil(n_total_rows / batch_size)))
    logger.info('')

    # iter trainset
    for i in tqdm(range(0, n_total_rows, batch_size)):
        row_idx_s = i  # start row's index for batch
        row_idx_e = i + batch_size  # end row's index for batch

        # for last iter
        if row_idx_e >= n_total_rows:
            row_idx_e = n_total_rows

        file_read_idx_s = math.floor(
            row_idx_s / n_rows_per_file)  # file index which contains start row index (aka start file)
        file_read_idx_e = math.ceil(
            row_idx_e / n_rows_per_file)  # file index which contains end row index (aka end file)

        rows_read_idx_s = row_idx_s % n_rows_per_file  # start row index on start file
        rows_read_idx_e = row_idx_e % n_rows_per_file  # end row index on end file

        train_X, train_Y, train_M = None, None, None

        # read files for batch
        for j in range(file_read_idx_s, file_read_idx_e):
            read_npz = np.load(os.path.join(preprocessed_dir, train_files[j]))

            if j == file_read_idx_s:
                train_X = read_npz['X'][rows_read_idx_s:]
                train_Y = read_npz['Y'][rows_read_idx_s:]
                train_M = read_npz['M'][rows_read_idx_s:]
            elif j == file_read_idx_e - 1:
                train_X = np.vstack((train_X, read_npz['X'][:rows_read_idx_e]))
                train_Y = np.vstack((train_Y, read_npz['Y'][:rows_read_idx_e]))
                train_M = np.vstack((train_M, read_npz['M'][:rows_read_idx_e]))
            else:
                train_X = np.vstack((train_X, read_npz['X']))
                train_Y = np.vstack((train_Y, read_npz['Y']))
                train_M = np.vstack((train_M, read_npz['M']))

        train_X, train_Y, train_M = train_X.reshape(-1, 10, 8), train_Y.reshape(-1, 8), train_M.reshape(-1, 77, 8)

        # # log
        # logger.info('X : {}, {}, {}'.format(train_X.shape, valid_X.shape, test_X.shape))
        # logger.info('Y : {}, {}, {}'.format(train_Y.shape, valid_Y.shape, test_Y.shape))
        # logger.info('M : {}, {}, {}'.format(train_M.shape, valid_M.shape, test_M.shape))
        # logger.info('Feed data : X{}, Y{}, M{}'.format(train_X.shape, train_Y.shape, train_M.shape))

        # return current batch
        yield train_X, train_Y, train_M, valid_X, valid_Y, valid_M, test_X, test_Y, test_M

##
# Legacy util.py functions

def get_logger(log_path='logs/', log_level=logging.INFO):
    """
    :param log_path
    :return: logger instance
    """
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logger = logging.getLogger()

    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s %(message)s', date_format)
    i = 0
    today = datetime.datetime.now()
    name = 'log-' + today.strftime('%Y%m%d') + '-' + '%02d' % i + '.log'
    while os.path.exists(log_path + name):
        i += 1
        name = 'log-' + today.strftime('%Y%m%d') + '-' + '%02d' % i + '.log'

    fileHandler = logging.FileHandler(os.path.join(log_path + name))
    streamHandler = logging.StreamHandler()

    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    logger.setLevel(log_level)
    logger.info('Writing logs at {}'.format(os.path.join(log_path + name)))

    return logger, log_path + name


def make_date_dir(path):
    """
    :param path
    :return: os.path.join(path+date_dir)
    """
    if not os.path.exists(path):
        os.mkdir(path)
    i = 0
    today = datetime.datetime.now()
    name = today.strftime('%Y%m%d') + '-' + '%02d' % i
    while os.path.exists(os.path.join(path + name)):
        i += 1
        name = today.strftime('%Y%m%d') + '-' + '%02d' % i
    os.mkdir(os.path.join(path + name))
    return os.path.join(path + name)


def find_latest_dir(path):
    dirs = os.listdir(path)
    dirs_splited = list(map(lambda x: x.split("-"), dirs))

    # find latest date
    dirs_date = [int(dir[0]) for dir in dirs_splited]
    dirs_date.sort()
    latest_date = dirs_date[-1]

    # find latest num in lastest date
    dirs_num = [int(dir[1]) for dir in dirs_splited if int(dir[0]) == latest_date]
    dirs_num.sort()
    latest_num = dirs_num[-1]
    latest_dir = str(latest_date) + '-' + '%02d' % latest_num

    return os.path.join(path + latest_dir)
