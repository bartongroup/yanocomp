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


def resample(arr, max_size):
    if len(arr) <= max_size:
        return arr
    else:
        return np.random.choice(arr, size=max_size, replace=False)


def kl_divergence(mvg_1, mvg_2, n_samples=10_000):
    '''monte carlo simulated KL divergence'''
    X = mvg_1.sample(n_samples)
    pred_1 = mvg_1.log_probability(X)
    pred_2 = mvg_2.log_probability(X)
    return pred_1.mean() - pred_2.mean()


def fit_multisamp_gmm(X, n_components=2, max_iterations=1e8, stop_threshold=0.1,
                      inertia=0.01, lr_decay=1e-3, pseudocount=0.5):
    '''
    Fit a gaussian mixture model to multiple samples. Each sample has its own
    GMM with its own weights but shares distributions with others. Returns
    a dists and weights both for combined data and for each sample
    '''
    samp_sizes = [len(samp) for samp in X]
    X_pooled = np.concatenate(X)
    n_dim = X_pooled.shape[1]
    kmeans = pm.Kmeans(n_components)
    kmeans.fit(X_pooled)
    init_pred = [
        kmeans.predict(samp) for samp in X
    ]
    init_pred_pooled = np.concatenate(init_pred)
    dists = [
        pm.IndependentComponentsDistribution([
            pm.NormalDistribution.from_samples(X_pooled[init_pred_pooled == i, j])
            for j in range(n_dim)])
        for i in range(n_components)   
    ]
    weights = [
        np.array([(samp_y == i).mean() for i in range(n_components)])
        for samp_y in init_pred
    ]
    # dists are shared, weights are not
    gmms = [pm.GeneralMixtureModel(dists, w) for w in weights]
    initial_log_probability_sum = -np.inf
    iteration, improvement = 0, np.inf
    while improvement > stop_threshold and iteration < max_iterations + 1:
        step_size = 1 - ((1 - inertia) * (2 + iteration) ** -lr_decay)
        if iteration:
            for d in dists:
                d.from_summaries(step_size)
            for gmm in gmms:
                if gmm.summaries.sum():
                    summaries = gmm.summaries + pseudocount
                    summaries /= summaries.sum()
                    gmm.weights[:] = np.log(summaries)
                    gmm.summaries[:] = 0

        log_probability_sum = 0
        for gmm, samp in zip(gmms, X):
            log_probability_sum += gmm.summarize(samp)

        if iteration == 0:
            initial_log_probability_sum = log_probability_sum
        else:
            improvement = log_probability_sum - last_log_probability_sum

        iteration += 1
        last_log_probability_sum = log_probability_sum

    per_samp_weights = np.array([np.exp(gmm.weights) for gmm in gmms])
    combined_weights = np.average(per_samp_weights, axis=0, weights=samp_sizes)
    return dists, combined_weights, per_samp_weights


def fit_gmm(cntrl, treat, expected_params,
            max_gmm_fit_depth=2000, min_mod_vs_unmod_kld=0.5):
    '''
    Fits a multivariate, two-gaussian GMM to data and measures KL
    divergence of the resulting distributions.
    '''
    n_cntrl = len(cntrl)
    pooled = cntrl + treat
    samp_sizes = np.array([len(samp) for samp in pooled])
    pooled = [
        resample(samp, max_gmm_fit_depth) for samp in pooled
    ]
    dists, weights, per_samp_weights = fit_multisamp_gmm(pooled, n_components=2)
    preds = np.round(per_samp_weights * samp_sizes[:, np.newaxis])
    kld = kl_divergence(dists[0], dists[1])

    if kld >= min_mod_vs_unmod_kld:
        # assess which dist is closer to expectation:
        expected = pm.IndependentComponentsDistribution([
            pm.NormalDistribution(*e) for e in expected_params
        ])
        dist_from_expected = [kl_divergence(expected, icd) for icd in dists]
        sort_idx = np.argsort(dist_from_expected)
        expected_kld = min(dist_from_expected)
        # sort so that null model is zero
        dists = [dists[i] for i in sort_idx]
        weights = weights[sort_idx]
        preds = preds[:, sort_idx]
    else:
        expected_kld = np.inf

    gmm = pm.GeneralMixtureModel(dists, weights)
    cntrl_preds, treat_preds = preds[:n_cntrl], preds[n_cntrl:]
    return gmm, kld, expected_kld, cntrl_preds, treat_preds


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


def gmm_g_test(cntrl_preds, treat_preds, p_val_threshold=0.05):
    '''
    Perform G tests for differential modification and within-condition
    homogeneity
    '''
    het_g, p_val = two_cond_g_test([
        cntrl_preds.sum(0), treat_preds.sum(0)
    ])
    if p_val < p_val_threshold:
        cntrl_hom_g, _, = two_cond_g_test(cntrl_preds)
        treat_hom_g, _, = two_cond_g_test(treat_preds)
        hom_g = cntrl_hom_g + treat_hom_g
        if hom_g >= het_g:
            p_val = 1
    else:
        hom_g = np.nan
    return het_g, hom_g, p_val


def calculate_mod_stats(cntrl_preds, treat_preds, pseudocount=0.5):
    cntrl_pred = cntrl_preds.sum(0)
    treat_pred = treat_preds.sum(0)
    cntrl_frac_mod = (cntrl_pred[1] + pseudocount) / (cntrl_pred.sum() + pseudocount)
    treat_frac_mod = (treat_pred[1] + pseudocount) / (treat_pred.sum() + pseudocount)
    log_odds = np.log2(treat_frac_mod) - np.log2(cntrl_frac_mod)
    return cntrl_frac_mod, treat_frac_mod, log_odds


def format_sm_preds(sm_preds, events):
    reps = events.index.get_level_values('replicate').tolist()
    read_ids = events.index.get_level_values('read_idx').tolist()
    events = events.values.tolist()
    sm_preds = sm_preds.tolist()
    sm_preds_dict = {}
    for r_id, rep, ev, p in zip(read_ids, reps, events, sm_preds):
        try:
            sm_preds_dict[rep]['read_ids'].append(r_id)
        except KeyError:
            sm_preds_dict[rep] = {
                'read_ids': [r_id,],
                'events': [],
                'preds': []
            }
        sm_preds_dict[rep]['events'].append(ev)
        sm_preds_dict[rep]['preds'].append(p)
    return sm_preds_dict


def position_stats(cntrl, treat, kmers,
                   max_gmm_fit_depth=2000,
                   max_cntrl_vs_exp_kld=1,
                   min_mod_vs_unmod_kld=0.5,
                   p_val_threshold=0.05,
                   model=load_model_priors(),
                   generate_sm_preds=False):
    '''
    Fits the GMM, estimates mod rates/changes, and performs G test
    '''
    window_size = len(kmers)
    centre = window_size // 2
    # first test that there is actually some difference in cntrl/treat
    # easiest way to do this is to just test the central kmer...
    pass_kstest = False
    pass_kld = False
    ks, ks_p_val = stats.ks_2samp(
        cntrl['mean'].values[:, centre],
        treat['mean'].values[:, centre],
    )
    # if there is we can perform the GMM fit and subsequent G test
    if ks_p_val < p_val_threshold:
        pass_kstest = True
        expected_params = model.loc[:, kmers]
        expected_params = expected_params.values.reshape(2, -1).T
        cntrl_fit_data = [c.values for _, c in cntrl.groupby('replicate')]
        treat_fit_data = [t.values for _, t in treat.groupby('replicate')]
        gmm, kld, exp_kld, cntrl_preds, treat_preds = fit_gmm(
            cntrl_fit_data, treat_fit_data, expected_params,
            max_gmm_fit_depth, min_mod_vs_unmod_kld
        )
        current_mean, current_std = gmm.distributions[1][centre].parameters
        cntrl_frac_mod, treat_frac_mod, log_odds = calculate_mod_stats(cntrl_preds, treat_preds)
        # if the KL divergence of the distributions is too small we stop here
        if kld >= min_mod_vs_unmod_kld and exp_kld <= max_cntrl_vs_exp_kld and kld > exp_kld:
            pass_kld = True
            g_stat, hom_g_stat, p_val = gmm_g_test(
                cntrl_preds, treat_preds,
                p_val_threshold=p_val_threshold
            )
    
    if pass_kld & pass_kstest:
        kmer = kmers[centre]
        # sort out the single molecule predictions
        if p_val < p_val_threshold and generate_sm_preds:
            cntrl_probs = gmm.predict_proba(np.concatenate(cntrl_fit_data))[:, 1]
            treat_probs = gmm.predict_proba(np.concatenate(treat_fit_data))[:, 1]
            sm_preds = {
                'kmers' : kmers.tolist(),
                'cntrl' : format_sm_preds(cntrl_probs, cntrl),
                'treat' : format_sm_preds(treat_probs, treat),
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
    depth = (events['mean'].notnull()
                           .groupby('replicate')
                           .sum())
    at_min_depth = (depth >= min_depth).all(0)
    valid_pos = at_min_depth.loc[at_min_depth].index.values
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
                              min_depth=5, window_size=3):
    cntrl_valid_pos = get_valid_pos(cntrl_events, min_depth)
    treat_valid_pos = get_valid_pos(treat_events, min_depth)
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


def test_depth(cntrl_pos_events, treat_pos_events, min_depth=10):
    cntrl_depth = cntrl_pos_events.groupby(level='replicate').size()
    treat_depth = treat_pos_events.groupby(level='replicate').size()
    return ((cntrl_depth >= min_depth).all() and 
            (treat_depth >= min_depth).all())


def create_positional_data(cntrl_events, treat_events, kmers, locus_id,
                           reverse, min_depth=5, window_size=3):
    # first attempt to filter out any positions below the min_depth threshold
    # we still need to check again later, this just prevents costly indexing ops...
    valid_pos = get_cntrl_treat_valid_pos(
        cntrl_events, treat_events, reverse,
        min_depth=min_depth,
        window_size=window_size,
    )
    for pos, win in valid_pos:
        cntrl_pos_events = index_pos_range(cntrl_events, win)
        treat_pos_events = index_pos_range(treat_events, win)
        if test_depth(cntrl_pos_events, treat_pos_events, min_depth):
            pos_kmers = kmers.loc[win].values
            yield (pos, locus_id, pos_kmers, cntrl_pos_events, treat_pos_events)


def iter_positions(gene_id, cntrl_datasets, treat_datasets, reverse,
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
                reverse,
                min_depth=min_depth,
                window_size=window_size
            )
    else:
        yield from create_positional_data(
            cntrl_events, treat_events, kmers,
            gene_id, reverse,
            min_depth=min_depth,
            window_size=window_size
        )


def test_chunk(gene_ids, cntrl_fns, treat_fns, model_fn=None, test_level='gene',
               window_size=3, min_depth=5, max_gmm_fit_depth=2000,
               max_cntrl_vs_exp_kld=1, min_mod_vs_unmod_kld=0.5,
               p_val_threshold=0.05, generate_sm_preds=False):
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
                reverse=True if strand == '-' else False,
                test_level=test_level,
                window_size=window_size,
                min_depth=min_depth
            )
            for pos, feature_id, kmers, cntrl, treat in pos_iter:
                r, sm = position_stats(
                    cntrl, treat, kmers,
                    max_gmm_fit_depth=max_gmm_fit_depth,
                    max_cntrl_vs_exp_kld=1,
                    min_mod_vs_unmod_kld=0.5,
                    p_val_threshold=p_val_threshold,
                    model=model,
                    generate_sm_preds=generate_sm_preds
                )
                if r[0]:
                    r = [chrom, pos, feature_id, strand] + r[1:]
                    chunk_res.append(r)
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
                  window_size=3, min_depth=5, max_gmm_fit_depth=2000,
                  max_cntrl_vs_exp_kld=1, min_mod_vs_unmod_kld=0.5,
                  p_val_threshold=0.05, processes=1,
                  generate_sm_preds=False):
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
        max_cntrl_vs_exp_kld=max_cntrl_vs_exp_kld,
        min_mod_vs_unmod_kld=min_mod_vs_unmod_kld,
        p_val_threshold=p_val_threshold,
        generate_sm_preds=generate_sm_preds,
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
        if p is not None:
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
@click.option('-d', '--max-gmm-fit-depth', required=False, default=2000)
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
    generate_sm_preds = output_sm_preds_fn is not None
    if test_gene is None:
        res, sm_preds = parallel_test(
            cntrl_hdf5_fns, treat_hdf5_fns,
            model_fn=prior_model_fn,
            test_level=test_level,
            window_size=window_size,
            min_depth=min_depth,
            max_gmm_fit_depth=max_gmm_fit_depth,
            max_cntrl_vs_exp_kld=1, # TODO
            min_mod_vs_unmod_kld=min_kl_divergence,
            p_val_threshold=fdr_threshold,
            processes=processes,
            generate_sm_preds=generate_sm_preds,
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
            max_cntrl_vs_exp_kld=1,
            min_mod_vs_unmod_kld=min_kl_divergence,
            p_val_threshold=fdr_threshold,
            generate_sm_preds=generate_sm_preds,
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