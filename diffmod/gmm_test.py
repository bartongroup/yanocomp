import os
import logging
from functools import partial
from contextlib import contextmanager
import multiprocessing as mp

import numpy as np
import pandas as pd
from pandas.core.computation.ops import UndefinedVariableError
from scipy import stats
from statsmodels.stats.multitest import multipletests
import h5py as h5
import pomegranate as pm

import click


logger = logging.getLogger('gmmtest')


RESULTS_COLUMNS = [
    'chrom', 'pos', 'gene_id', 'kmer', 'strand',
    'log_odds', 'p_val', 'fdr',
    'cntrl_frac_mod', 'treat_frac_mod',
    'g_stat', 'hom_g_stat', 'kl_divergence',
]


def get_model(model_fn=None):
    '''
    Load the parameters for the expected kmer distributions.
    '''
    if model_fn is None:
        model_fn = os.path.join(
            os.path.split(os.path.abspath(__file__))[0],
            'data/vir1_r9.4_model.tsv'
        )
    m = pd.read_csv(
        model_fn, sep='\t', comment='#', index_col='kmer'
    )
    m = m[['current_mean', 'current_std', 'dwell_mean', 'dwell_std']]
    return m.T.to_dict(orient='list')


def check_custom_filter_exprs(exprs, columns=RESULTS_COLUMNS):
    '''
    checks filter expressions for pandas dataframes using dummy dataframe
    '''
    dummy = pd.DataFrame([], columns=RESULTS_COLUMNS)
    try:
        dummy.filter(exprs)
    except:
        raise ValueError(f'custom filter expression "{exprs}" is invalid, check docs')


@contextmanager
def hdf5_list(hdf5_fns):
    '''Context manager for list of HDF5 files'''
    hdf5_list = [
        h5.File(fn, 'r') for fn in hdf5_fns
    ]
    try:
        yield hdf5_list
    finally:
        for f in hdf5_list:
            f.close()


def get_expressed_genes(datasets):
    '''
    Identify the genes which have eventaligned reads in all samples
    '''
    # filter out any transcripts that are not expressed in all samples
    genes = set(datasets[0].keys())
    for d in datasets[1:]:
        genes.intersection_update(set(d.keys()))
    return list(genes)


def fetch_gene_events(gene_id, datasets,
                      fetch_transcript_ids=False,
                      fetch_read_ids=False):
    '''
    Extract the event alignment table for a given gene from a
    list of HDF5 file objects
    '''
    gene_events = []
    for i, d in enumerate(datasets, 1):
        # read full dataset from disk
        e = pd.DataFrame(d[f'{gene_id}/events'][:])
        # convert f16 to f64
        e['mean'] = e['mean'].astype(np.float64)
        e['duration'] = np.log10(e['duration'].astype(np.float64))
        if fetch_transcript_ids:
            t = d[f'{gene_id}/transcript_ids'][:]
            t = {i: t_id for i, t_id in enumerate(t)}
            e['transcript_idx'] = e.transcript_idx.map(t)
        if fetch_read_ids:
            r = d[f'{gene_id}/read_ids'][:].astype('U32')
            r = {i: r_id for i, r_id in enumerate(r)}
            e['read_idx'] = e.read_idx.map(r)
        e['replicate'] = i
        gene_events.append(e)
    return pd.concat(gene_events)


def fetch_gene_attrs(gene_id, datasets):
    '''
    Extracts important info i.e. chromosome, strand and kmers
    for a gene from the HDF5 files...
    '''
    kmers = {}
    # get general info which should be same for all datasets
    g = datasets[0][gene_id]
    chrom = g.attrs['chrom']
    strand = g.attrs['strand']
    # positions (and their kmers) which are recorded may vary across datasets
    for d in datasets:
        k = d[f'{gene_id}/kmers'][:].astype(
            np.dtype([('pos', np.uint32), ('kmer', 'U5')])
        )
        kmers.update(dict(k))
    return kmers, chrom, strand


def iter_positions(gene_id, cntrl_datasets, treat_datasets,
                   min_depth=10):
    '''
    Generator which iterates over the positions in a gene
    which have the minimum depth in eventaligned reads.
    '''
    cntrl_events = fetch_gene_events(gene_id, cntrl_datasets)
    treat_events = fetch_gene_events(gene_id, treat_datasets)
    kmers, chrom, strand = fetch_gene_attrs(
        gene_id, cntrl_datasets + treat_datasets
    )
    for pos, kmer in kmers.items():
        cntrl_pos_events = cntrl_events.query(
            f'pos == {pos}', parser='pandas', engine='numexpr'
        )
        treat_pos_events = treat_events.query(
            f'pos == {pos}', parser='pandas', engine='numexpr'
        )
        cntrl_depth = cntrl_pos_events.replicate.value_counts()
        treat_depth = treat_pos_events.replicate.value_counts()
        if (cntrl_depth >= min_depth).all() and (treat_depth >= min_depth).all():
            yield chrom, pos, strand, kmer, cntrl_pos_events, treat_pos_events


def kl_divergence(X_mu, X_sigma, Y_mu, Y_sigma):
    '''
    measure of divergence between two distributions
    '''
    X_var, Y_var = X_sigma ** 2, Y_sigma ** 2
    lsr = np.log(Y_sigma / X_sigma)
    sdm = (X_mu - Y_mu) ** 2
    return  lsr + (X_var + sdm) / (2 * Y_var) - 0.5


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


def fit_gmm(cntrl, treat, expected_params, max_gmm_fit_depth=10_000,
            balance_method=None):
    '''
    Fits a bivariate, two-gaussian GMM to pooled data and measures KL divergence of
    the resulting distributions.
    '''
    cntrl, treat = correct_class_imbalance(
        cntrl, treat, max_gmm_fit_depth, balance_method
    )
    pooled = np.concatenate([
        cntrl[['mean', 'duration']].values,
        treat[['mean', 'duration']].values
    ])
    assert len(pooled) <= max_gmm_fit_depth
    # Model starts as two identical bi-variate normal distributions initialised from
    # the expected parameters. The first is frozen (i.e. cannot be fit).
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
    # just use the kl divergence for the currents cause I don't understnad
    # how to calculate for multivariate gaussians...
    kld = kl_divergence(
        *gmm.distributions[0].parameters[0][0].parameters,
        *gmm.distributions[1].parameters[0][0].parameters,
    )
    return gmm, kld


def gmm_predictions(gmm, events, pseudocount=np.finfo('f').tiny):
    '''
    Use the GMM to make single molecule predictions and calculate fraction modified.
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
    Perform G tests for differential modification and within-condition homogeneity
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
                   min_kld=0.5,
                   p_val_threshold=0.05,
                   model=get_model()):
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
            max_gmm_fit_depth, balance_method
        )
        fit_mean, fit_std = gmm.distributions[1].parameters
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
            True, log_odds, p_val, 1, # placeholder for fdr
            cntrl_frac_mod, treat_frac_mod,
            g_stat, hom_g_stat, kld
        ]
    else:
        return [False,]


def test_chunk(gene_ids, cntrl_fns, treat_fns, model_fn=None,
               min_depth=10, max_gmm_fit_depth=10_000, balance_method=None,
               min_kld=0.5, p_val_threshold=0.05):
    '''
    run the GMM tests on a subset of gene_ids
    '''
    model = get_model(model_fn)
    chunk_res = []
    with hdf5_list(cntrl_fns) as cntrl_h5, hdf5_list(treat_fns) as treat_h5:
        for gene_id in gene_ids:
            pos_iter = iter_positions(gene_id, cntrl_h5, treat_h5, min_depth)
            for chrom, pos, strand, kmer, cntrl, treat in pos_iter:
                r = position_stats(
                    cntrl, treat, kmer,
                    max_gmm_fit_depth=max_gmm_fit_depth,
                    balance_method=balance_method,
                    min_kld=min_kld,
                    p_val_threshold=p_val_threshold,
                    model=model,
                )
                if r[0]:
                    r = [chrom, pos, gene_id, kmer, strand] + r[1:]
                    chunk_res.append(r)
    return chunk_res


def parallel_test(cntrl_fns, treat_fns, model_fn=None,
                  min_depth=10, max_gmm_fit_depth=10_000, balance_method=None,
                  min_kld=0.5, p_val_threshold=0.05, processes=1):
    '''
    Runs the GMM tests on positions from gene_ids which are found in all HDF5.
    Gene ids are processed as parallel chunks.
    '''
    with hdf5_list(cntrl_fns) as cntrl_h5, hdf5_list(treat_fns) as treat_h5:
        gene_ids = sorted(get_expressed_genes(cntrl_h5 + treat_h5))[:120]
        gene_id_chunks = np.array_split(gene_ids, processes)

    logger.info(f'{len(gene_ids):,} genes to be processed on {processes} workers')
    with mp.Pool(processes) as pool:
        _test_chunk = partial(
            test_chunk,
            cntrl_fns=cntrl_fns,
            treat_fns=treat_fns,
            model_fn=model_fn,
            min_depth=min_depth,
            max_gmm_fit_depth=max_gmm_fit_depth,
            balance_method=balance_method,
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


def to_bed(res, output_bed_fn, fdr_threshold=0.05, custom_filter=None):
    '''
    write main results to bed file
    '''
    sig_res = res.query(f'fdr < {fdr_threshold}')
    logger.info(
        f'{len(sig_res):,} positions significant at {fdr_threshold * 100:.0f}% level'
    )
    if custom_filter is not None:
        sig_res = sig_res.query(custom_filter)
        logger.info(f'{len(sig_res):,} positions pass filter "{custom_filter}"')
    sig_res = sig_res.sort_values(by=['chrom', 'pos'])
    logger.info(f'Writing output to {os.path.abspath(output_bed_fn)}')
    with open(output_bed_fn, 'w') as bed:
        for record in sig_res.itertuples(index=False):
            (chrom, pos, gene_id, kmer, strand,
             log_odds, pval, fdr, c_fm, t_fm,
             g_stat, hom_g_stat, kld) = record
            nlogfdr = - int(round(np.log10(fdr)))
            bed_record = (
                f'{chrom:s}\t{pos - 2:d}\t{pos + 3:d}\t'
                f'{gene_id}:{kmer}\t{nlogfdr:d}\t{strand:s}\t'
                f'{log_odds:.2f}\t{pval:.2g}\t{fdr:.2g}\t'
                f'{c_fm:.2f}\t{t_fm:.2f}\t'
                f'{g_stat:.2f}\t{hom_g_stat:.2f}\t{kld:.2f}\n'
            )
            bed.write(bed_record)


@click.command()
@click.option('-c', '--cntrl-hdf5-fns', required=True, multiple=True)
@click.option('-t', '--treat-hdf5-fns', required=True, multiple=True)
@click.option('-o', '--output-bed-fn', required=True)
@click.option('-m', '--prior-model-fn', required=False, default=None)
@click.option('-n', '--min-depth', required=False, default=10)
@click.option('-m', '--max-gmm-fit-depth', required=False, default=10_000)
@click.option('-b', '--class-balance-method', required=False,
              type=click.Choice(['none', 'undersample', 'oversample']), default='none')
@click.option('-k', '--min-kl-divergence', required=False, default=0.5)
@click.option('-f', '--fdr-threshold', required=False, default=0.05)
@click.option('--custom-filter', required=False, default=None)
@click.option('-p', '--processes', required=False, default=1)
def gmm_test(cntrl_hdf5_fns, treat_hdf5_fn, output_bed_fn, prior_model_fn,
             min_depth, max_gmm_fit_depth, class_balance_method,
             min_kl_divergence, fdr_threshold,
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
        min_depth=min_depth,
        max_gmm_fit_depth=max_gmm_fit_depth,
        balance_method=class_balance_method,
        min_kld=min_kl_divergence,
        p_val_threshold=fdr_threshold,
        processes=processes
    )
    to_bed(res, output_bed_fn, fdr_threshold, custom_filter)


if __name__ == '__main__':
    gmm_test()