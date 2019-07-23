import os
import logging

logger = None

def get_logger(configuration=None):
  if configuration is not None:
    global logger
    logger = create_logger(configuration)
    return logger
  else:
    return logger


def create_logger(configuration):
  result_dir = configuration.result_dir
  backtest_date = configuration.backtest_date

  logger = logging.getLogger('logger')
  log_format = '%(asctime)s - %(levelname)-3s - %(message)s'
  time_format = "%Y-%m-%d %H:%M:%S"
  formatter = logging.Formatter(log_format, time_format)

  log_dir = os.path.join(result_dir, str(backtest_date))
  os.mkdir(log_dir)

  log_filepath = os.path.join(log_dir, 'log.txt')
  c_handler = logging.StreamHandler()
  f_handler = logging.FileHandler(log_filepath)
  c_handler.setFormatter(formatter)
  f_handler.setFormatter(formatter)

  logger.addHandler(c_handler)
  logger.addHandler(f_handler)
  logger.setLevel(logging.INFO)

  return logger