import logging
from functools import partial
import dataclasses
import multiprocessing as mp

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import click

from .io import (
    hdf5_list, get_shared_keys, load_model_priors,
    load_gene_kmers, load_gene_events, load_gene_attrs,
    save_gmmtest_results, save_sm_preds
)
from .stats import GMMTestResults, position_stats

logger = logging.getLogger('diffmod')


def get_valid_pos(events, min_read_depth):
    depth = (events['mean'].notnull()
                           .groupby('replicate')
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
    # bug #22797 in pandas makes this difficult
    #events = events.loc[:, pd.IndexSlice[:, win]]
    idx = pd.MultiIndex.from_arrays([
        np.repeat(['mean', 'duration'], len(win)),
        np.tile(win, 2)
    ])
    events = events.reindex(idx, axis=1)
    events = events.dropna(axis=0)
    return events


def test_depth(cntrl_pos_events, treat_pos_events, min_read_depth=10):
    cntrl_depth = cntrl_pos_events.groupby(level='replicate').size()
    treat_depth = treat_pos_events.groupby(level='replicate').size()
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
    model = load_model_priors(opts.prior_model_fn)
    chunk_res = []
    chunk_sm_preds = {}

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
                    cntrl, treat, kmers,
                    max_fit_depth=opts.max_fit_depth,
                    max_cntrl_vs_exp_emd=opts.max_cntrl_vs_exp_emd,
                    min_mod_vs_unmod_emd=opts.min_mod_vs_unmod_emd,
                    p_val_threshold=opts.fdr_threshold,
                    model=model,
                    generate_sm_preds=opts.generate_sm_preds
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
    with hdf5_list(opts.cntrl_hdf5_fns) as cntrl_h5, \
         hdf5_list(opts.treat_hdf5_fns) as treat_h5:

        gene_ids = sorted(get_shared_keys(cntrl_h5 + treat_h5))
        gene_id_chunks = np.array_split(gene_ids, opts.processes * 4)

    logger.info(
        f'{len(gene_ids):,} genes to be processed on {opts.processes} workers'
    )
    if opts.processes > 1:
        with mp.Pool(opts.processes) as pool:
            res = []
            sm_preds = {}
            for chunk_res, chunk_sm_preds in pool.imap_unordered(
                    partial(test_chunk, opts), gene_id_chunks):
                res += chunk_res
                sm_preds.update(chunk_sm_preds)
    else:
        res, sm_preds = test_chunk(opts, gene_id_chunks[0])

    logger.info(f'Complete. Tested {len(res):,} positions')
    res = pd.DataFrame(res)
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


@dataclasses.dataclass
class GMMTestOpts:
    cntrl_hdf5_fns: str
    treat_hdf5_fns: str
    output_bed_fn: str
    output_sm_preds_fn: str = None
    prior_model_fn: str = None
    test_level: str = 'gene'
    window_size: int = 5
    min_read_depth: int = 5
    max_fit_depth: int = 1000
    max_cntrl_vs_exp_emd: float = 1 # needs a click option
    min_mod_vs_unmod_emd: float = 1
    fdr_threshold: float = 0.05
    processes: float = 1
    test_gene: str = None
    random_seed: int = None
    generate_sm_preds: bool = dataclasses.field(init=False)

    def __post_init__(self):
        if self.random_seed is None:
            self.random_seed = abs(hash(self.output_bed_fn))
        self.generate_sm_preds = self.output_sm_preds_fn is not None

        
def make_dataclass_decorator(dc):
    def _dataclass_decorator(cmd):
        @click.pass_context
        def _make_dataclass(ctx, **kwargs):
            return ctx.invoke(cmd, dc(**kwargs))
        return _make_dataclass
    return _dataclass_decorator


@click.command()
@click.option('-c', '--cntrl-hdf5-fns', required=True, multiple=True)
@click.option('-t', '--treat-hdf5-fns', required=True, multiple=True)
@click.option('-o', '--output-bed-fn', required=True)
@click.option('-s', '--output-sm-preds-fn', required=False, default=None)
@click.option('-m', '--prior-model-fn', required=False, default=None)
@click.option('--test-level', required=False, default='gene',
              type=click.Choice(['gene', 'transcript']))
@click.option('-w', '--window-size', required=False, default=5)
@click.option('-n', '--min-read-depth', required=False, default=5)
@click.option('-d', '--max-fit-depth', required=False, default=1000)
@click.option('-e', '--min-mod-vs-unmod_emd', required=False, default=1.)
@click.option('-f', '--fdr-threshold', required=False, default=0.05)
@click.option('-p', '--processes', required=False,
              default=1, type=click.IntRange(1, None))
@click.option('--test-gene', required=False, default=None, hidden=True)
@click.option('--random-seed', required=False, default=None, hidden=True)
@make_dataclass_decorator(GMMTestOpts)
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
            f'Testing single gene {opts.test_gene}'
        )
        res, sm_preds = test_chunk(opts, [opts.test_gene])
        res = pd.DataFrame(res)
        _, res['fdr'], _, _ = multipletests(res.p_val, method='fdr_bh')

    res, sm_preds = filter_results(res, sm_preds, opts.fdr_threshold)

    save_gmmtest_results(res, opts.output_bed_fn)
    if opts.generate_sm_preds:
        save_sm_preds(
            sm_preds,
            opts.cntrl_hdf5_fns, opts.treat_hdf5_fns,
            opts.output_sm_preds_fn,
        )


if __name__ == '__main__':
    gmm_test()