import logging


def Logger(log_path=None, terminal=True):
    logger = logging.getLogger()
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    #
    if terminal:
        log_handler_terminal = logging.StreamHandler()
        log_handler_terminal.setFormatter(log_formatter)
        logger.addHandler(log_handler_terminal)

    if log_path is not None:
        log_handler_file = logging.FileHandler(log_path)
        log_handler_file.setFormatter(log_formatter)
        logger.addHandler(log_handler_file)

    logger.setLevel(logging.INFO)
    return logger
