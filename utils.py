import logging
import os

from settings import PROJECT_ROOT


def get_logger(level=logging.DEBUG, name="root"):
    os.chdir(PROJECT_ROOT)
    log = logging.getLogger()
    logger_path = os.path.join(PROJECT_ROOT, 'configs', os.environ.get("LOGGER_CONFIG_NAME", 'log_config.ini'))
    # try:
    #     logging.config.fileConfig(logger_path, disable_existing_loggers=False)
    # except Exception as ex:
    #     print(str(ex))
    if not log.handlers:
        log.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        try:
            from colorlog import ColoredFormatter
            formatter = ColoredFormatter(
                '%(log_color)s%(filename)s %(funcName)s  %(message)s%(reset)s')
            ch.setFormatter(formatter)
        except ModuleNotFoundError:
            pass
        log.addHandler(ch)
    return log