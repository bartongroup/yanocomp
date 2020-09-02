import click

from .nanopolish_collapse import nanopolish_collapse
from .gmm_test import gmm_test


@click.group()
def cli():
    pass

cli.add_command(nanopolish_collapse, name='prep')
cli.add_command(gmm_test, name='test')


if __name__ == '__main__':    
    cli()