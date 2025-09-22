import logging
import paddle
# 配置 logging 输出到控制台
def get_logger(log_level, name="root"):
    logger = logging.getLogger(name)

    # Avoid printing multiple logs
    logger.propagate = False

    if not logger.handlers:
        log_handler = logging.StreamHandler()
        logger.setLevel(log_level)
        log_format = logging.Formatter(
            '[%(asctime)-15s] [%(levelname)8s] %(filename)s:%(lineno)s - %(message)s'
        )
        log_handler.setFormatter(log_format)
        logger.addHandler(log_handler)
    else:
        logger.setLevel(log_level)
    return logger
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = get_logger("INFO","minimind")
# 测试不同级别的日志输出
# logging.debug('This is a debug message')
# logging.info('This is an info message')
# logging.warning('This is a warning message')
# logging.error('This is an error message')
# logging.critical('This is a critical message')
logger.info('Logger is configured and ready to use.')
