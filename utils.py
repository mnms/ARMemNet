import logging
import logging.handlers
import os
import datetime
from mpi4py import MPI

def get_logger(log_path='logs/'):
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
    name = 'log-'+today.strftime('%Y%m%d')+'-'+'%02d'%i+'.log'
    while os.path.exists(log_path+name):
        i += 1
        name = 'log-'+today.strftime('%Y%m%d')+'-'+'%02d'%i+'.log'
    
    fileHandler = logging.FileHandler(os.path.join(log_path+name))
    streamHandler = logging.StreamHandler()
    
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)
    
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    
    logger.setLevel(logging.INFO)
    logger.info('Writing logs at {}'.format(os.path.join(log_path+name)))
    return logger, log_path+name

def get_mpi_info():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    host = comm.Get_attr(MPI.HOST)
    univ_sz= comm.Get_attr(MPI.UNIVERSE_SIZE)
    return size, rank, host, univ_sz


def make_date_dir(path):
    """
    :param path
    :return: os.path.join(path+date_dir)
    """
    if not os.path.exists(path):
        os.mkdir(path)
    i = 0
    today = datetime.datetime.now()
    name = today.strftime('%Y%m%d')+'-'+'%02d' % i
    while os.path.exists(os.path.join(path + name)):
        i += 1
        name = today.strftime('%Y%m%d')+'-'+'%02d' % i
    os.mkdir(os.path.join(path + name))
    return os.path.join(path + name)

def find_latest_dir(path):
    dirs = os.listdir(path)
    dirs_splited = list(map(lambda x:x.split("-"), dirs))
    
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

def write_latest_dir_to_cache(cache_file, latest_checkpoint_dir):
    with open(cache_file, "w") as f:
        f.write(latest_checkpoint_dir)#.encode("utf-8"))
        f.close()

def read_latest_dir_from_cache(cache_file):
    if os.path.isfile(cache_file) is False:
        return None

    with open(cache_file, "r") as f:
        latest_dir = f.read()#.decode("utf-8")
        f.close()
        return latest_dir
