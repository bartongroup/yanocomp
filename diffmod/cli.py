import cProfile
import pstats
from io import StringIO
import atexit
import logging
import click
import click_log

from .prep import nanopolish_collapse
from .priors import model_priors
from .gmmtest import gmm_test


COMMANDS = {
    'prep': nanopolish_collapse,
    'priors': model_priors,
    'gmmtest': gmm_test,
}


def get_logger(name):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '{asctime} {levelname:8s} {message}', style='{'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = get_logger('diffmod')


@click.group(commands=COMMANDS)
@click_log.simple_verbosity_option(logger)
@click.option("--profile", is_flag=True, hidden=True)
def cli(profile):
    if profile:
        logger.debug('Running with profiler...')
        prof = cProfile.Profile()
        prof.enable()

        @atexit.register
        def print_profile_stats_on_exit():
            prof.disable()
            logger.debug('Profiling complete')
            s = StringIO()
            prof_stats = pstats.Stats(prof, stream=s)
            prof_stats.sort_stats('cumulative').print_stats(50)
            logger.debug(s.getvalue())


if __name__ == '__main__':
    cli()