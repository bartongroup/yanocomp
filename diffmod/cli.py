import logging
import click
import click_log

from .prep import nanopolish_collapse
from .priors import model_priors
from .gmmtest import gmm_test


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


log_opt = get_logger('diffmod')

COMMANDS = {
    'prep': log_opt(nanopolish_collapse),
    'priors': log_opt(model_priors),
    'gmmtest': log_opt(gmm_test),
}


@click.group(commands=COMMANDS)
def cli():
    pass


if __name__ == '__main__':
    cli()