import logging
from functools import partial
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pomegranate as pm

import click

from .io import (
    hdf5_list, get_shared_keys, load_model_priors,
    load_gene_kmers, load_gene_events, load_gene_attrs,
    save_gmmtest_results, save_sm_preds
)

logger = logging.getLogger('diffmod')


def kl_divergence(mvg_1, mvg_2, n_samples=10_000):
    '''monte carlo simulated KL divergence'''
    X = mvg_1.sample(n_samples)
    pred_1 = mvg_1.log_probability(X)
    pred_2 = mvg_2.log_probability(X)
    return pred_1.mean() - pred_2.mean()


def correct_class_imbalance(cntrl, treat, max_depth, method=None):
    n_cntrl, n_treat = len(cntrl), len(treat)
    max_depth_per_samp = max_depth // 2
    rng = np.random.default_rng()
    if method is None or method == 'none':
        if n_cntrl > max_depth_per_samp:
            cntrl = rng.choice(cntrl, max_depth_per_samp, replace=False)
        if n_treat > max_depth_per_samp:
            treat = rng.choice(treat, max_depth_per_samp, replace=False)
    elif method == 'undersample':
        n_samp = min(n_treat, n_cntrl, max_depth_per_samp)
        cntrl = rng.choice(cntrl, n_samp, replace=False)
        treat = rng.choice(treat, n_samp, replace=False)
    elif method == 'oversample':
        n_samp = min(max_depth_per_samp, max(n_cntrl, n_treat))
        cntrl = rng.choice(cntrl, n_samp, replace=True)
        treat = rng.choice(treat, n_samp, replace=True)
    else:
        raise ValueError(f'Sampling method "{method}" not implemented')
    return cntrl, treat


def remove_outliers(gmm, X, quantile=0.99):
    prob = gmm.probability(X)
    perc = np.quantile(prob, 1 - quantile)
    return X[prob > perc]


def fit_gmm(cntrl, treat, expected_params,
            max_gmm_fit_depth=10_000,
            balance_method=None, refit_quantile=0.95):
    '''
    Fits a multivariate, two-gaussian GMM to pooled data and measures KL
    divergence of the resulting distributions.
    '''
    window_size = len(expected_params) // 2
    centre = window_size // 2
    cntrl, treat = correct_class_imbalance(
        cntrl, treat, max_gmm_fit_depth, balance_method
    )
    pooled = np.concatenate([cntrl, treat], axis=0)
    # Model starts as two identical multivariate normal distributions initialised
    # from the expected parameters. The first is frozen (i.e. cannot be fit).
    gmm = pm.GeneralMixtureModel([
        pm.IndependentComponentsDistribution([
            pm.NormalDistribution(*p, frozen=True) for p in expected_params
        ], frozen=True),
        pm.IndependentComponentsDistribution([
            pm.NormalDistribution(*p) for p in expected_params
        ]),
    ])
    gmm.fit(pooled)
    # sometimes using the first model to remove outliers can improve the fit
    if refit_quantile != 1:
        pooled = remove_outliers(gmm, pooled, refit_quantile)
        gmm.fit(pooled)
    kld = kl_divergence(gmm.distributions[0], gmm.distributions[1])
    return gmm, kld


def gmm_predictions(gmm, events, pseudocount=np.finfo('f').tiny):
    '''
    Use the GMM to make single molecule predictions and calculate fraction
    modified.
    '''
    probs = gmm.predict_proba(events)
    preds = probs.argmax(1)
    log_prob = np.log2(probs.sum(0) + pseudocount)
    log_ratio = log_prob[1] - log_prob[0]
    frac_mod = preds.sum() / len(preds)
    return preds, probs[:, 1], frac_mod, log_ratio


def crosstab_preds_per_replicate(preds, replicates):
    '''
    Makes contingency table of replicates vs predictions
    '''
    # make categorical to catch cases where 100% of preds are one class
    preds = pd.Categorical(preds, categories=[0, 1])
    ct = pd.crosstab(replicates, preds, dropna=False).values
    return ct


def two_cond_g_test(counts):
    '''
    G test, catch errors caused by rows/columns with all zeros
    '''
    try:
        g_stat, p_val, *_ = stats.chi2_contingency(
            counts, lambda_='log-likelihood'
        )
        return g_stat, p_val
    except ValueError:
        # one cond is empty
        return 0.0, 1.0


def gmm_g_test(cntrl_preds, cntrl_reps,
               treat_preds, treat_reps,
               p_val_threshold=0.05):
    '''
    Perform G tests for differential modification and within-condition
    homogeneity
    '''
    cntrl_counts = crosstab_preds_per_replicate(cntrl_preds, cntrl_reps)
    treat_counts = crosstab_preds_per_replicate(treat_preds, treat_reps)
    het_g, p_val = two_cond_g_test([
        cntrl_counts.sum(0), treat_counts.sum(0)
    ])
    if p_val < p_val_threshold:
        cntrl_hom_g, _, = two_cond_g_test(cntrl_counts)
        treat_hom_g, _, = two_cond_g_test(treat_counts)
        hom_g = cntrl_hom_g + treat_hom_g
        if hom_g >= het_g:
            p_val = 1
    else:
        hom_g = np.nan
    return het_g, hom_g, p_val


def format_sm_preds(sm_preds, index):
    reps = index.get_level_values('replicate').values
    read_ids = index.get_level_values('read_idx').values
    sm_preds_dict = {}
    for rep, read_id, p in zip(reps, read_ids, sm_preds):
        rep = int(rep)
        p = float(p)
        try:
            sm_preds_dict[rep][read_id] = p
        except KeyError:
            sm_preds_dict[rep] = {}
            sm_preds_dict[rep][read_id] = p
    return sm_preds_dict


def position_stats(cntrl, treat, kmers,
                   max_gmm_fit_depth=10_000,
                   balance_method=None,
                   refit_quantile=0.95,
                   min_kld=0.5,
                   p_val_threshold=0.05,
                   model=load_model_priors()):
    '''
    Fits the GMM, estimates mod rates/changes, and performs G test
    '''
    window_size = len(kmers)
    centre = window_size // 2
    # first test that there is actually some difference in cntrl/treat
    # easiest way to do this is to just test the central kmer...
    pass_ttest = False
    pass_kld = False
    _, tt_p_val = stats.ttest_ind(
        cntrl.values[:, centre],
        treat.values[:, centre],
        equal_var=False
    )
    # if there is we can perform the GMM fit and subsequent G test
    if tt_p_val < p_val_threshold:
        pass_ttest = True
        expected_params = model.loc[kmers].values
        cntrl_fit_data = cntrl.values
        treat_fit_data = treat.values
        gmm, kld = fit_gmm(
            cntrl_fit_data, treat_fit_data,
            expected_params,
            max_gmm_fit_depth, balance_method,
            refit_quantile
        )
        current_mean, current_std = gmm.distributions[1][centre].parameters
        # if the KL divergence of the distributions is too small we stop here
        if kld >= min_kld:
            pass_kld = True
            cntrl_preds, cntrl_probs, cntrl_frac_mod, cntrl_lr = gmm_predictions(
                gmm, cntrl_fit_data
            )
            treat_preds, treat_probs, treat_frac_mod, treat_lr = gmm_predictions(
                gmm, treat_fit_data
            )
            log_odds = treat_lr - cntrl_lr
            g_stat, hom_g_stat, p_val = gmm_g_test(
                cntrl_preds, cntrl.index.get_level_values('replicate').values,
                treat_preds, treat.index.get_level_values('replicate').values,
                p_val_threshold=p_val_threshold
            )
    if pass_kld & pass_ttest:
        kmer = kmers[centre]
        # sort out the single molecule predictions
        if p_val < p_val_threshold:
            sm_preds = {
                'cntrl' : format_sm_preds(cntrl_probs, cntrl.index),
                'treat' : format_sm_preds(treat_probs, treat.index),
            }
        else:
            sm_preds = None
        return [
            True, kmer, log_odds, p_val, 1,       # placeholder for fdr
            cntrl_frac_mod, treat_frac_mod,
            g_stat, hom_g_stat,
            current_mean, current_std, kld
        ], sm_preds
    else:
        return [False, ], None


def get_valid_pos(events, min_depth):
    depth = (events.notnull()
                   .groupby('replicate')
                   .sum())
    at_min_depth = (depth >= min_depth).all(0)
    valid_pos = at_min_depth.loc[at_min_depth].index.values
    return set(valid_pos)


def get_valid_windows(valid_pos, window_size=3):
    w = window_size // 2
    for pos in valid_pos:
        win = np.arange(pos - w, pos + w + 1)
        if valid_pos.issuperset(win):
            yield pos, win


def get_cntrl_treat_valid_pos(cntrl_events, treat_events,
                              min_depth=5, window_size=3):
    cntrl_valid_pos = get_valid_pos(cntrl_events, min_depth)
    treat_valid_pos = get_valid_pos(treat_events, min_depth)
    valid_pos = cntrl_valid_pos.intersection(treat_valid_pos)
    yield from get_valid_windows(valid_pos, window_size)  


def index_pos_range(events, win):
    events = events.loc[:, win]
    events = events.dropna(axis=0)
    return events


def test_depth(cntrl_pos_events, treat_pos_events, min_depth=10):
    cntrl_depth = cntrl_pos_events.groupby(level='replicate').size()
    treat_depth = treat_pos_events.groupby(level='replicate').size()
    return ((cntrl_depth >= min_depth).all() and 
            (treat_depth >= min_depth).all())


def create_positional_data(cntrl_events, treat_events, kmers, locus_id,
                           min_depth=5, window_size=3):
    # first attempt to filter out any positions below the min_depth threshold
    # we still need to check again later, this just prevents costly indexing ops...
    valid_pos = get_cntrl_treat_valid_pos(
        cntrl_events, treat_events,
        min_depth=min_depth,
        window_size=window_size,
    )
    for pos, win in valid_pos:
        cntrl_pos_events = index_pos_range(cntrl_events, win)
        treat_pos_events = index_pos_range(treat_events, win)
        if test_depth(cntrl_pos_events, treat_pos_events, min_depth):
            pos_kmers = kmers.loc[win].values
            yield (pos, locus_id, pos_kmers, cntrl_pos_events, treat_pos_events)


def iter_positions(gene_id, cntrl_datasets, treat_datasets,
                   test_level='gene', window_size=3, min_depth=5):
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
                min_depth=min_depth,
                window_size=window_size
            )
    else:
        yield from create_positional_data(
            cntrl_events, treat_events, kmers,
            gene_id, min_depth=min_depth,
            window_size=window_size
        )


def test_chunk(gene_ids, cntrl_fns, treat_fns, model_fn=None, test_level='gene',
               window_size=3, min_depth=5, max_gmm_fit_depth=10_000,
               balance_method=None, refit_quantile=0.95, min_kld=0.5,
               p_val_threshold=0.05):
    '''
    run the GMM tests on a subset of gene_ids
    '''
    model = load_model_priors(model_fn)
    chunk_res = []
    chunk_sm_preds = {}
    with hdf5_list(cntrl_fns) as cntrl_h5, hdf5_list(treat_fns) as treat_h5:
        for gene_id in gene_ids:
            chrom, strand = load_gene_attrs(gene_id, cntrl_h5)
            pos_iter = iter_positions(
                gene_id, cntrl_h5, treat_h5,
                test_level=test_level,
                window_size=window_size,
                min_depth=min_depth
            )
            for pos, feature_id, kmers, cntrl, treat in pos_iter:
                r, sm = position_stats(
                    cntrl, treat, kmers,
                    max_gmm_fit_depth=max_gmm_fit_depth,
                    balance_method=balance_method,
                    refit_quantile=refit_quantile,
                    min_kld=min_kld,
                    p_val_threshold=p_val_threshold,
                    model=model,
                )
                if r[0]:
                    r = [chrom, pos, feature_id, strand] + r[1:]
                    chunk_res.append(r)
                    if sm is not None:
                        pos = int(pos)
                        try:
                            chunk_sm_preds[feature_id][pos] = sm
                        except KeyError:
                            chunk_sm_preds[feature_id] = {}
                            chunk_sm_preds[feature_id][pos] = sm
    return chunk_res, chunk_sm_preds


RESULTS_COLUMNS = [
    'chrom', 'pos', 'gene_id', 'strand', 'kmer',
    'log_odds', 'p_val', 'fdr',
    'cntrl_frac_mod', 'treat_frac_mod',
    'g_stat', 'hom_g_stat',
    'current_mu', 'current_std', 'kl_divergence',
]


def parallel_test(cntrl_fns, treat_fns, model_fn=None, test_level='gene',
                  window_size=3, min_depth=5, max_gmm_fit_depth=10_000,
                  balance_method=None, refit_quantile=0.95, min_kld=0.5,
                  p_val_threshold=0.05, processes=1):
    '''
    Runs the GMM tests on positions from gene_ids which are found in all HDF5.
    Gene ids are processed as parallel chunks.
    '''
    with hdf5_list(cntrl_fns) as cntrl_h5, hdf5_list(treat_fns) as treat_h5:
        gene_ids = sorted(get_shared_keys(cntrl_h5 + treat_h5))
        gene_id_chunks = np.array_split(gene_ids, processes)

    logger.info(
        f'{len(gene_ids):,} genes to be processed on {processes} workers'
    )
    _test_chunk = partial(
        test_chunk,
        cntrl_fns=cntrl_fns,
        treat_fns=treat_fns,
        model_fn=model_fn,
        test_level=test_level,
        window_size=window_size,
        min_depth=min_depth,
        max_gmm_fit_depth=max_gmm_fit_depth,
        balance_method=balance_method,
        refit_quantile=refit_quantile,
        min_kld=min_kld,
        p_val_threshold=p_val_threshold
    )
    if processes > 1:
        with mp.Pool(processes) as pool:
            res = []
            sm_preds = {}
            for chunk_res, chunk_sm_preds in pool.imap_unordered(
                    _test_chunk, gene_id_chunks):
                res += chunk_res
                sm_preds.update(chunk_sm_preds)
    else:
        res, sm_preds = _test_chunk(gene_id_chunks[0])

    logger.info(f'Complete. Tested {len(res):,} positions')
    res = pd.DataFrame(res, columns=RESULTS_COLUMNS)
    _, res['fdr'], _, _ = multipletests(res.p_val, method='fdr_bh')
    return res, sm_preds


def filter_results(res, sm_preds, fdr_threshold, custom_filter):
    sig_res = res.query(f'fdr < {fdr_threshold}')
    logger.info(
        f'{len(sig_res):,} positions significant at '
        f'{fdr_threshold * 100:.0f}% level'
    )
    if custom_filter is not None:
        sig_res = sig_res.query(custom_filter)
        logger.info(
            f'{len(sig_res):,} positions pass filter "{custom_filter}"'
        )
    sig_sm_preds = {}
    for gene_id, pos in sig_res[['gene_id', 'pos']].itertuples(index=False):
        p = sm_preds[gene_id][pos]
        try:
            sig_sm_preds[gene_id][pos] = p
        except KeyError:
            sig_sm_preds[gene_id] = {}
            sig_sm_preds[gene_id][pos] = p
    return sig_res, sig_sm_preds


def check_custom_filter_exprs(ctx, param, exprs):
    '''
    checks filter expressions for pandas dataframes using dummy dataframe
    '''
    if exprs is not None:
        dummy = pd.DataFrame([], columns=RESULTS_COLUMNS)
        # if expression is incorrect this will fail
        try:
            dummy.query(exprs)
        except Exception as e:
            raise click.BadParameter(f'--custom-filter issue: {str(e)}')


def set_default_kl_divergence(ctx, param, val):
    if val is None:
        win_size = ctx.params['window_size']
        if win_size == 1:
            val = 0.5
        else:
            val = 2
        logger.warn(f'Default min kl divergence set to {val} to match '
                    f'window size {win_size}')
    return val


@click.command()
@click.option('-c', '--cntrl-hdf5-fns', required=True, multiple=True)
@click.option('-t', '--treat-hdf5-fns', required=True, multiple=True)
@click.option('-o', '--output-bed-fn', required=True)
@click.option('-s', '--output-sm-preds-fn', required=False, default=None)
@click.option('-m', '--prior-model-fn', required=False, default=None)
@click.option('--test-level', required=False, default='gene',
              type=click.Choice(['gene', 'transcript']))
@click.option('-w', '--window-size', required=False, default=3)
@click.option('-n', '--min-depth', required=False, default=5)
@click.option('-d', '--max-gmm-fit-depth', required=False, default=10_000)
@click.option('-b', '--class-balance-method', required=False,
              type=click.Choice(['none', 'undersample', 'oversample']),
              default='none')
@click.option('-q', '--outlier-quantile', required=False, default=0.95)
@click.option('-k', '--min-kl-divergence', required=False,
              default=None, callback=set_default_kl_divergence)
@click.option('-f', '--fdr-threshold', required=False, default=0.05)
@click.option('--custom-filter', required=False, default=None,
              callback=check_custom_filter_exprs)
@click.option('-p', '--processes', required=False,
              default=1, type=click.IntRange(1, None))
@click.option('--test-gene', required=False, default=None, hidden=True)
def gmm_test(cntrl_hdf5_fns, treat_hdf5_fns, output_bed_fn, output_sm_preds_fn,
             prior_model_fn, test_level, window_size, min_depth,
             max_gmm_fit_depth, class_balance_method, outlier_quantile,
             min_kl_divergence, fdr_threshold, custom_filter,
             processes, test_gene):
    '''
    Differential RNA modifications using nanopore DRS signal level data
    '''
    logger.info(
        f'Running gmmtest with {len(cntrl_hdf5_fns):,} control '
        f'datasets and {len(treat_hdf5_fns):,} treatment datasets'
    )
    if test_gene is None:
        res, sm_preds = parallel_test(
            cntrl_hdf5_fns, treat_hdf5_fns,
            model_fn=prior_model_fn,
            test_level=test_level,
            window_size=window_size,
            min_depth=min_depth,
            max_gmm_fit_depth=max_gmm_fit_depth,
            balance_method=class_balance_method,
            refit_quantile=outlier_quantile,
            min_kld=min_kl_divergence,
            p_val_threshold=fdr_threshold,
            processes=processes
        )
    else:
        logger.info(
            f'Testing single gene {test_gene}'
        )
        res, sm_preds = test_chunk(
            [test_gene],
            cntrl_hdf5_fns, treat_hdf5_fns,
            model_fn=prior_model_fn,
            test_level=test_level,
            window_size=window_size,
            min_depth=min_depth,
            max_gmm_fit_depth=max_gmm_fit_depth,
            balance_method=class_balance_method,
            refit_quantile=outlier_quantile,
            min_kld=min_kl_divergence,
            p_val_threshold=fdr_threshold,
        )
        res = pd.DataFrame(res, columns=RESULTS_COLUMNS)
        _, res['fdr'], _, _ = multipletests(res.p_val, method='fdr_bh')

    res, sm_preds = filter_results(
        res, sm_preds, fdr_threshold, custom_filter
    )

    save_gmmtest_results(
        res, output_bed_fn,
    )
    if output_sm_preds_fn is not None:
        save_sm_preds(
            sm_preds,
            cntrl_hdf5_fns, treat_hdf5_fns,
            output_sm_preds_fn,
        )


if __name__ == '__main__':
    gmm_test()