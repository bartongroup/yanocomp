from functools import wraps
import logging
import click
import click_log

from .nanopolish_collapse import nanopolish_collapse
from .gmm_test import gmm_test


def get_logger(name):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '{asctime} {levelname:8s} {message}', style='{'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    log_opt = click_log.simple_verbosity_option(logger)
    return log_opt


COMMANDS = {
    'prep': get_logger('prep')(nanopolish_collapse),
    'gmmtest': get_logger('gmmtest')(gmm_test),
}


@click.group(commands=COMMANDS)
def cli():
    pass



if __name__ == '__main__':    
    cli()