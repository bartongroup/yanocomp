import logging
from functools import partial
import dataclasses
import multiprocessing as mp

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import click

from .opts import make_dataclass_decorator, dynamic_dataclass
from .io import (
    hdf5_list, get_shared_keys, load_model_priors,
    load_gene_kmers, load_gene_events, load_gene_attrs,
    save_gmmtest_results, save_sm_preds
)
from .stats import GMMTestResults, position_stats, assign_modified_distribution


logger = logging.getLogger('yanocomp')


def get_valid_pos(events, min_read_depth):
    depth = (events.notnull()
                   .groupby('replicate', sort=False)
                   .sum())
    at_min_read_depth = (depth >= min_read_depth).all(0)
    valid_pos = at_min_read_depth.loc[at_min_read_depth].index.values
    return set(valid_pos)


def get_valid_windows(valid_pos, reverse, window_size=3):
    w = window_size // 2
    for pos in valid_pos:
        win = np.arange(pos - w, pos + w + 1)
        if reverse:
            win = win[::-1]
        if valid_pos.issuperset(win):
            yield pos, win


def get_cntrl_treat_valid_pos(cntrl_events, treat_events, reverse,
                              min_read_depth=5, window_size=3):
    cntrl_valid_pos = get_valid_pos(cntrl_events, min_read_depth)
    treat_valid_pos = get_valid_pos(treat_events, min_read_depth)
    valid_pos = cntrl_valid_pos.intersection(treat_valid_pos)
    yield from get_valid_windows(valid_pos, reverse, window_size)  


def index_pos_range(events, win):
    events = events.loc[:, win]
    events = events.dropna(axis=0)
    return events


def test_depth(cntrl_pos_events, treat_pos_events, min_read_depth=10):
    cntrl_depth = cntrl_pos_events.groupby(level='replicate', sort=False).size()
    treat_depth = treat_pos_events.groupby(level='replicate', sort=False).size()
    return ((cntrl_depth >= min_read_depth).all() and 
            (treat_depth >= min_read_depth).all())


def create_positional_data(cntrl_events, treat_events, kmers, locus_id,
                           reverse, min_read_depth=5, window_size=3):
    # first attempt to filter out any positions below the min_read_depth threshold
    # we still need to check again later, this just prevents costly indexing ops...
    valid_pos = get_cntrl_treat_valid_pos(
        cntrl_events, treat_events, reverse,
        min_read_depth=min_read_depth,
        window_size=window_size,
    )
    for pos, win in valid_pos:
        cntrl_pos_events = index_pos_range(cntrl_events, win)
        treat_pos_events = index_pos_range(treat_events, win)
        if test_depth(cntrl_pos_events, treat_pos_events, min_read_depth):
            pos_kmers = kmers.loc[win].values
            yield (pos, locus_id, pos_kmers, cntrl_pos_events, treat_pos_events)


def iter_positions(gene_id, cntrl_datasets, treat_datasets, reverse,
                   test_level='gene', window_size=3, min_read_depth=5):
    '''
    Generator which iterates over the positions in a gene
    which have the minimum depth in eventaligned reads.
    '''
    by_transcript = test_level == 'transcript'
    cntrl_events = load_gene_events(
        gene_id, cntrl_datasets,
        by_transcript_ids=by_transcript
    )
    treat_events = load_gene_events(
        gene_id, treat_datasets,
        by_transcript_ids=by_transcript
    )
    kmers = load_gene_kmers(
        gene_id, cntrl_datasets + treat_datasets
    )
    if by_transcript:
        # events are dicts of dataframes
        valid_transcripts = set(cntrl_events).intersection(treat_events)
        for transcript_id in valid_transcripts:
            yield from create_positional_data(
                cntrl_events[transcript_id],
                treat_events[transcript_id], 
                kmers, transcript_id,
                reverse,
                min_read_depth=min_read_depth,
                window_size=window_size
            )
    else:
        yield from create_positional_data(
            cntrl_events, treat_events, kmers,
            gene_id, reverse,
            min_read_depth=min_read_depth,
            window_size=window_size
        )


@dataclasses.dataclass
class PosRecord:
    chrom: str
    pos: int
    gene_id: str
    strand: str


@dataclasses.dataclass
class GMMTestRecord(GMMTestResults, PosRecord):

    @staticmethod
    def from_records(chrom, pos, feature_id, strand, gmmtest_results):
        return GMMTestRecord(
            chrom, pos, feature_id, strand,
            **dataclasses.asdict(gmmtest_results)
        )


def test_chunk(opts, gene_ids):
    '''
    run the GMM tests on a subset of gene_ids
    '''
    # convert back to dataclass (have to convert to dict for pickling)
    opts = dynamic_dataclass('GMMTestOpts', bases=(GMMTestOpts,), **opts)
    chunk_res = []
    chunk_sm_preds = {}

    gene_ids, random_seed = gene_ids
    random_state = np.random.default_rng(random_seed)

    with hdf5_list(opts.cntrl_hdf5_fns) as cntrl_h5, \
         hdf5_list(opts.treat_hdf5_fns) as treat_h5:

        for gene_id in gene_ids:
            chrom, strand = load_gene_attrs(gene_id, cntrl_h5)
            pos_iter = iter_positions(
                gene_id, cntrl_h5, treat_h5,
                reverse=True if strand == '-' else False,
                test_level=opts.test_level,
                window_size=opts.window_size,
                min_read_depth=opts.min_read_depth
            )
            for pos, feature_id, kmers, cntrl, treat in pos_iter:
                was_tested, result, sm = position_stats(
                    cntrl, treat, kmers, opts,
                    random_state=random_state
                )
                if was_tested:
                    record = GMMTestRecord.from_records(
                        chrom, pos, feature_id, strand, result
                    )
                    chunk_res.append(record)
                    pos = int(pos)
                    try:
                        chunk_sm_preds[feature_id][pos] = sm
                    except KeyError:
                        chunk_sm_preds[feature_id] = {}
                        chunk_sm_preds[feature_id][pos] = sm
    return chunk_res, chunk_sm_preds


def parallel_test(opts):
    '''
    Runs the GMM tests on positions from gene_ids which are found in all HDF5.
    Gene ids are processed as parallel chunks.
    '''
    # use seed sequence for processes
    ss = np.random.SeedSequence(opts.random_seed)
    random_seeds = list(ss.spawn(opts.processes))

    with hdf5_list(opts.cntrl_hdf5_fns) as cntrl_h5, \
         hdf5_list(opts.treat_hdf5_fns) as treat_h5:

        gene_ids = sorted(get_shared_keys(cntrl_h5 + treat_h5))
        gene_id_chunks = np.array_split(gene_ids, opts.processes)

    logger.info(
        f'{len(gene_ids):,} genes to be processed on {opts.processes} workers'
    )
    if opts.processes > 1:
        with mp.Pool(opts.processes) as pool:
            res = []
            sm_preds = {}
            for chunk_res, chunk_sm_preds in pool.imap_unordered(
                    partial(test_chunk, dataclasses.asdict(opts)),
                    zip(gene_id_chunks, random_seeds)):
                res += chunk_res
                sm_preds.update(chunk_sm_preds)
    else:
        res, sm_preds = test_chunk(
            dataclasses.asdict(opts),
            (gene_id_chunks[0], opts.random_seed)
        )

    logger.info(f'Complete. Tested {len(res):,} positions')
    res = pd.DataFrame(
        res,
        columns=list(PosRecord.__annotations__.keys()) + \
                list(GMMTestResults.__annotations__.keys())
    )
    res.dropna(subset=['p_val'], inplace=True) # TODO not sure how some nans creep in
    if len(res):
        _, res['fdr'], _, _ = multipletests(res.p_val, method='fdr_bh')
    return res, sm_preds


def filter_results(res, sm_preds, fdr_threshold):
    sig_res = res.query(f'fdr < {fdr_threshold}')
    logger.info(
        f'{len(sig_res):,} positions significant at '
        f'{fdr_threshold * 100:.0f}% level'
    )
    sig_sm_preds = {}
    for gene_id, pos in sig_res[['gene_id', 'pos']].itertuples(index=False):
        p = sm_preds[gene_id][pos]
        if p is not None:
            try:
                sig_sm_preds[gene_id][pos] = p
            except KeyError:
                sig_sm_preds[gene_id] = {}
                sig_sm_preds[gene_id][pos] = p
    return sig_res, sig_sm_preds


def set_default_depth(ctx, param, val):
    if val is None:
        win_size = ctx.params['window_size']
        val = max(win_size * 2, 5) # at least as many reads per sample as features
        logger.warn(f'Default min depth set to {val} to match '
                    f'window size {win_size}')
    return val


@dataclasses.dataclass
class GMMTestOpts:
    generate_sm_preds: bool = dataclasses.field(init=False)

    def __post_init__(self):
        if self.random_seed is None:
            self.random_seed = len(self.output_bed_fn)

        self.generate_sm_preds = self.output_sm_preds_fn is not None
        self.model = load_model_priors(self.model_fn)


@click.command(options_metavar='''-c <cntrl_hdf5_1> \\
                                  -c <cntrl_hdf5_2> \\
                                  -t <treat_hdf5_1> \\
                                  -t <treat_hdf5_2> \\
                                  [OPTIONS]''',
              short_help='Differential RNA mod analysis')
@click.option('-c', '--cntrl-hdf5-fns', required=True, multiple=True,
              help='Control HDF5 files. Can specify multiple files using multiple -c flags')
@click.option('-t', '--treat-hdf5-fns', required=True, multiple=True,
              help='Treatment HDF5 files. Can specify multiple files using multiple -t flags')
@click.option('-o', '--output-bed-fn', required=True, help='Output bed file name')
@click.option('-s', '--output-sm-preds-fn', required=False, default=None,
              help='JSON file to output single molecule predictions. Can be gzipped (detected from name)')
@click.option('-m', '--model-fn', required=False, default=None,
              help='Model file with expected kmer current distributions')
@click.option('--test-level', required=False, default='gene',
              type=click.Choice(['gene', 'transcript']), show_default=True,
              help='Test at transcript level or aggregate to gene level')
@click.option('-w', '--window-size', required=False, default=5, show_default=True,
              help='How many adjacent kmers to model over')
@click.option('-u', '--add-uniform/--no-uniform', required=False, default=True, hidden=True,
              help=('Whether to include a uniform component in GMMs to detect outliers caused by '
                    'alignment errors. Helps to improve the robustness of the modelling'))
@click.option('-e', '--outlier-factor', required=False, default=0.5, show_default=True,
              help=('Scaling factor for labelling outliers during model initialisation. '
                    'Smaller means more aggressive labelling of outliers'))
@click.option('-n', '--min-read-depth', required=False, default=None, type=int,
              callback=set_default_depth,
              help='Minimum reads per replicate to test a position. Default is to set dynamically')
@click.option('-d', '--max-fit-depth', required=False, default=1000, show_default=True,
              help='Maximum number of reads per replicate used to fit the model')
@click.option('-k', '--min-ks', required=False, default=0.2, show_default=True,
              help='Minimum KS test statistic to attempt to build a model for a position')
@click.option('-f', '--fdr-threshold', required=False, default=0.05, show_default=True,
              help='False discovery rate threshold for output')
@click.option('-p', '--processes', required=False, show_default=True,
              default=1, type=click.IntRange(1, None))
@click.option('--test-gene', required=False, default=None, hidden=True, multiple=True)
@click.option('--random-seed', required=False, default=None, hidden=True)
@make_dataclass_decorator('GMMTestOpts', bases=(GMMTestOpts,))
def gmm_test(opts):
    '''
    Differential RNA modifications using nanopore DRS signal level data
    '''
    logger.info(
        f'Running gmmtest with {len(opts.cntrl_hdf5_fns):,} control '
        f'datasets and {len(opts.treat_hdf5_fns):,} treatment datasets'
    )
    if opts.test_gene is None:
        res, sm_preds = parallel_test(opts)
    else:
        logger.info(
            f'Testing gene {" ".join(opts.test_gene)}'
        )
        res, sm_preds = test_chunk(
            dataclasses.asdict(opts),
            (opts.test_gene, opts.random_seed)
        )
        res = pd.DataFrame(
            res,
            columns=list(PosRecord.__annotations__.keys()) + \
                    list(GMMTestResults.__annotations__.keys())
        )
        logger.info(f'Complete. Tested {len(res):,} positions')
        res.dropna(subset=['p_val'], inplace=True)
        if len(res):
            _, res['fdr'], _, _ = multipletests(res.p_val, method='fdr_bh')

    res, sm_preds = assign_modified_distribution(
        *filter_results(res, sm_preds, opts.fdr_threshold), opts.model
    )

    save_gmmtest_results(res, opts.output_bed_fn)
    if opts.generate_sm_preds:
        save_sm_preds(
            sm_preds,
            opts.cntrl_hdf5_fns, opts.treat_hdf5_fns,
            opts.output_sm_preds_fn,
        )


if __name__ == '__main__':
    gmm_test()