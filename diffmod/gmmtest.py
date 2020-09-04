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
    save_gmmtest_results,
)

logger = logging.getLogger('diffmod')


def kl_divergence(X_mu, X_sigma, Y_mu, Y_sigma):
    '''
    measure of divergence between two distributions
    '''
    X_var, Y_var = X_sigma ** 2, Y_sigma ** 2
    lsr = np.log(Y_sigma / X_sigma)
    sdm = (X_mu - Y_mu) ** 2
    return lsr + (X_var + sdm) / (2 * Y_var) - 0.5


def correct_class_imbalance(cntrl, treat, max_depth, method=None):
    n_cntrl, n_treat = len(cntrl), len(treat)
    max_depth_per_samp = max_depth // 2
    if method is None or method == 'none':
        if n_cntrl > max_depth_per_samp:
            cntrl = cntrl.sample(max_depth_per_samp, replace=False)
        if n_treat > max_depth_per_samp:
            treat = treat.sample(max_depth_per_samp, replace=False)
    elif method == 'undersample':
        n_samp = min(n_treat, n_cntrl, max_depth_per_samp)
        cntrl = cntrl.sample(n_samp, replace=False)
        treat = treat.sample(n_samp, replace=False)
    elif method == 'oversample':
        n_samp = min(max_depth_per_samp, max(n_cntrl, n_treat))
        cntrl = cntrl.sample(n_samp, replace=True)
        treat = treat.sample(n_samp, replace=True)
    else:
        raise ValueError(f'Sampling method "{method}" not implemented')
    return cntrl, treat


def remove_outliers(gmm, X, quantile=0.99):
    prob = gmm.probability(X)
    perc = np.quantile(prob, 1 - quantile)
    return X[prob > perc]


def fit_gmm(cntrl, treat, expected_params, max_gmm_fit_depth=10_000,
            balance_method=None, refit_quantile=0.95):
    '''
    Fits a bivariate, two-gaussian GMM to pooled data and measures KL
    divergence of the resulting distributions.
    '''
    cntrl, treat = correct_class_imbalance(
        cntrl, treat, max_gmm_fit_depth, balance_method
    )
    pooled = np.concatenate([
        cntrl[['mean', 'duration']].values,
        treat[['mean', 'duration']].values
    ])
    assert len(pooled) <= max_gmm_fit_depth
    # Model starts as two identical bi-variate normal distributions initialised
    # from the expected parameters. The first is frozen (i.e. cannot be fit).
    current_mu, current_sigma, dwell_mu, dwell_sigma = expected_params
    gmm = pm.GeneralMixtureModel([
        pm.IndependentComponentsDistribution([
            pm.NormalDistribution(current_mu, current_sigma),
            pm.NormalDistribution(dwell_mu, dwell_sigma),
        ], frozen=True),
        pm.IndependentComponentsDistribution([
            pm.NormalDistribution(current_mu, current_sigma),
            pm.NormalDistribution(dwell_mu, dwell_sigma),
        ]),
    ])
    gmm.fit(pooled)
    # sometimes using the first model to remove outliers can improve the fit
    if refit_quantile != 1:
        pooled = remove_outliers(gmm, pooled, refit_quantile)
        gmm.fit(pooled)
    # just use the kl divergence for the currents cause I don't understnad
    # how to calculate for multivariate gaussians...
    kld = kl_divergence(
        *gmm.distributions[0][0].parameters,
        *gmm.distributions[1][0].parameters,
    )
    return gmm, kld


def gmm_predictions(gmm, events, pseudocount=np.finfo('f').tiny):
    '''
    Use the GMM to make single molecule predictions and calculate fraction
    modified.
    '''
    probs = gmm.predict_proba(events[['mean', 'duration']])
    preds = probs.argmax(1)
    log_prob = np.log2(probs.sum(0) + pseudocount)
    log_ratio = log_prob[1] - log_prob[0]
    frac_mod = preds.sum() / len(preds)
    return preds, frac_mod, log_ratio


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


def position_stats(cntrl, treat, kmer,
                   max_gmm_fit_depth=10_000,
                   balance_method=None,
                   refit_quantile=0.95,
                   min_kld=0.5,
                   p_val_threshold=0.05,
                   model=load_model_priors()):
    '''
    Fits the GMM, estimates mod rates/changes, and performs G test
    '''
    # first test that there is actually some difference in cntrl/treat
    pass_ttest = False
    pass_kld = False
    _, tt_p_val = stats.ttest_ind(
        cntrl['mean'], treat['mean'], equal_var=False
    )
    # if there is we can perform the GMM fit and subsequent G test
    if tt_p_val < p_val_threshold:
        pass_ttest = True
        expected_params = model[kmer]
        gmm, kld = fit_gmm(
            cntrl, treat, expected_params,
            max_gmm_fit_depth, balance_method,
            refit_quantile
        )
        current_mean, current_std = gmm.distributions[1][0].parameters
        dwell_mean, dwell_std = gmm.distributions[1][1].parameters
        # if the KL divergence of the distributions is too small we stop here
        if kld >= min_kld:
            pass_kld = True
            cntrl_preds, cntrl_frac_mod, cntrl_lr = gmm_predictions(gmm, cntrl)
            treat_preds, treat_frac_mod, treat_lr = gmm_predictions(gmm, treat)
            log_odds = treat_lr - cntrl_lr
            g_stat, hom_g_stat, p_val = gmm_g_test(
                cntrl_preds, cntrl['replicate'],
                treat_preds, treat['replicate'],
                p_val_threshold=p_val_threshold
            )
    if pass_kld & pass_ttest:
        return [
            True, log_odds, p_val, 1,       # placeholder for fdr
            cntrl_frac_mod, treat_frac_mod,
            g_stat, hom_g_stat,
            current_mean, current_std,
            dwell_mean, dwell_std, kld
        ]
    else:
        return [False, ]


def test_depth(cntrl_pos_events, treat_pos_events, min_depth=10):
    cntrl_depth = cntrl_pos_events.replicate.value_counts()
    treat_depth = treat_pos_events.replicate.value_counts()
    return ((cntrl_depth >= min_depth).all() and 
            (treat_depth >= min_depth).all())


def iter_cntrl_treat_transcripts(cntrl, treat):
    transcript_ids = set(cntrl.transcript_idx).intersection(
        treat.transcript_idx
    )
    for transcript_id in transcript_ids:
        query = f'transcript_idx == "{transcript_id}"'
        yield transcript_id, cntrl.query(query), treat.query(query)


def iter_positions(gene_id, cntrl_datasets, treat_datasets,
                   test_level='gene', min_depth=10):
    '''
    Generator which iterates over the positions in a gene
    which have the minimum depth in eventaligned reads.
    '''
    cntrl_events = load_gene_events(
        gene_id, cntrl_datasets,
        load_transcript_ids=test_level == 'transcript'
    )
    treat_events = load_gene_events(
        gene_id, treat_datasets,
        load_transcript_ids=test_level == 'transcript'
    )

    kmers = load_gene_kmers(gene_id, cntrl_datasets + treat_datasets)
    chrom, strand = load_gene_attrs(
        gene_id, cntrl_datasets
    )
    for pos, kmer in kmers.items():
        cntrl_pos_events = cntrl_events.query(
            f'pos == {pos}', parser='pandas', engine='numexpr'
        )
        treat_pos_events = treat_events.query(
            f'pos == {pos}', parser='pandas', engine='numexpr'
        )
        if test_level == 'gene':
            if test_depth(cntrl_pos_events, treat_pos_events, min_depth):
                yield (
                    chrom, pos, gene_id, kmer, strand,
                    cntrl_pos_events, treat_pos_events
                )
        else:
            for transcript in iter_cntrl_treat_transcripts(cntrl_pos_events, 
                                                           treat_pos_events):
                transcript_id, cntrl_tpos_events, treat_tpos_events = transcript
                if test_depth(cntrl_tpos_events, treat_tpos_events, min_depth):
                    yield (
                        chrom, pos, transcript_id, kmer, strand,
                        cntrl_tpos_events, treat_tpos_events
                    )


def test_chunk(gene_ids, cntrl_fns, treat_fns, model_fn=None, test_level='gene',
               min_depth=10, max_gmm_fit_depth=10_000, balance_method=None,
               refit_quantile=0.95, min_kld=0.5, p_val_threshold=0.05):
    '''
    run the GMM tests on a subset of gene_ids
    '''
    model = load_model_priors(model_fn)
    chunk_res = []
    with hdf5_list(cntrl_fns) as cntrl_h5, hdf5_list(treat_fns) as treat_h5:
        for gene_id in gene_ids:
            pos_iter = iter_positions(
                gene_id, cntrl_h5, treat_h5, test_level, min_depth
            )
            for chrom, pos, feature_id, kmer, strand, cntrl, treat in pos_iter:
                r = position_stats(
                    cntrl, treat, kmer,
                    max_gmm_fit_depth=max_gmm_fit_depth,
                    balance_method=balance_method,
                    refit_quantile=refit_quantile,
                    min_kld=min_kld,
                    p_val_threshold=p_val_threshold,
                    model=model,
                )
                if r[0]:
                    r = [chrom, pos, feature_id, kmer, strand] + r[1:]
                    chunk_res.append(r)
    return chunk_res


RESULTS_COLUMNS = [
    'chrom', 'pos', 'gene_id', 'kmer', 'strand',
    'log_odds', 'p_val', 'fdr',
    'cntrl_frac_mod', 'treat_frac_mod',
    'g_stat', 'hom_g_stat',
    'current_mu', 'current_std', 'dwell_mu', 'dwell_std',
    'kl_divergence',
]


def parallel_test(cntrl_fns, treat_fns, model_fn=None, test_level='gene',
                  min_depth=10, max_gmm_fit_depth=10_000, balance_method=None,
                  refit_quantile=0.95, min_kld=0.5, p_val_threshold=0.05,
                  processes=1):
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
    with mp.Pool(processes) as pool:
        _test_chunk = partial(
            test_chunk,
            cntrl_fns=cntrl_fns,
            treat_fns=treat_fns,
            model_fn=model_fn,
            test_level=test_level,
            min_depth=min_depth,
            max_gmm_fit_depth=max_gmm_fit_depth,
            balance_method=balance_method,
            refit_quantile=refit_quantile,
            min_kld=min_kld,
            p_val_threshold=p_val_threshold
        )
        res = []
        for chunk_res in pool.imap_unordered(_test_chunk, gene_id_chunks):
            res += chunk_res

    logger.info(f'Complete. Tested {len(res):,} positions')
    res = pd.DataFrame(res, columns=RESULTS_COLUMNS)
    _, res['fdr'], _, _ = multipletests(res.p_val, method='fdr_bh')
    return res


def check_custom_filter_exprs(exprs, columns=RESULTS_COLUMNS):
    '''
    checks filter expressions for pandas dataframes using dummy dataframe
    '''
    dummy = pd.DataFrame([], columns=RESULTS_COLUMNS)
    # if expression is incorrect this will fail
    dummy.filter(exprs)


@click.command()
@click.option('-c', '--cntrl-hdf5-fns', required=True, multiple=True)
@click.option('-t', '--treat-hdf5-fns', required=True, multiple=True)
@click.option('-o', '--output-bed-fn', required=True)
@click.option('-m', '--prior-model-fn', required=False, default=None)
@click.option('--test-level', required=False, default='gene',
              type=click.Choice(['gene', 'transcript']))
@click.option('-n', '--min-depth', required=False, default=10)
@click.option('-d', '--max-gmm-fit-depth', required=False, default=10_000)
@click.option('-b', '--class-balance-method', required=False,
              type=click.Choice(['none', 'undersample', 'oversample']),
              default='none')
@click.option('-q', '--outlier-quantile', required=False, default=0.95)
@click.option('-k', '--min-kl-divergence', required=False, default=0.5)
@click.option('-f', '--fdr-threshold', required=False, default=0.05)
@click.option('--custom-filter', required=False, default=None)
@click.option('-p', '--processes', required=False, default=1)
def gmm_test(cntrl_hdf5_fns, treat_hdf5_fns, output_bed_fn, prior_model_fn,
             test_level, min_depth, max_gmm_fit_depth, class_balance_method,
             outlier_quantile, min_kl_divergence, fdr_threshold,
             custom_filter, processes):
    '''
    Differential RNA modifications using nanopore DRS signal level data
    '''
    if custom_filter is not None:
        # better to check now than fail after all that computation...
        check_custom_filter_exprs(custom_filter)
    np.seterr(under='ignore')
    logger.info(
        f'Running gmmtest with {len(cntrl_hdf5_fns):,} control '
        f'datasets and {len(treat_hdf5_fns):,} treatment datasets'
    )
    res = parallel_test(
        cntrl_hdf5_fns, treat_hdf5_fns,
        model_fn=prior_model_fn,
        test_level=test_level,
        min_depth=min_depth,
        max_gmm_fit_depth=max_gmm_fit_depth,
        balance_method=class_balance_method,
        refit_quantile=outlier_quantile,
        min_kld=min_kl_divergence,
        p_val_threshold=fdr_threshold,
        processes=processes
    )
    save_gmmtest_results(
        res, output_bed_fn,
        fdr_threshold, custom_filter
    )


if __name__ == '__main__':
    gmm_test()