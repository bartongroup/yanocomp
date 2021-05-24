import dataclasses

import numpy as np
import pandas as pd
from statsmodels.multivariate.pca import PCA
from scipy import stats
import pomegranate as pm

from .io import load_model_priors


N_COMPONENTS = 2


def pca_kstest(cntrl_data, treat_data):
    '''
    Transform multivariate data to univariate using PCA and perform
    Kolmogorov-Smirnov test
    '''
    n_cntrl = len(cntrl_data)
    pooled = np.concatenate([cntrl_data, treat_data])
    comps = PCA(pooled, 1).factors.ravel()
    ks, p_val = stats.ks_2samp(comps[:n_cntrl], comps[n_cntrl:])
    return ks, p_val


def median_absolute_deviation(arr):
    '''Returns the MAD of an array'''
    return np.median(np.abs(arr - np.median(arr)))


def kmeans_init_clusters(X, detect_outliers=True, init_method='first-k',
                         batch_size=100, max_iter=4, outlier_factor=0.5):
    '''
    Get predictions for initialising GMM using KMeans. Outliers are detected
    by calculating the MAD of the distances to the nearest centroid, and then
    labelling all values with a dist of > outlier_factor * MAD as outliers.
    '''
    kmeans = pm.Kmeans(N_COMPONENTS, init=init_method)
    kmeans.fit(X, batch_size=batch_size, max_iterations=max_iter)
    centroid_dists = kmeans.distance(X)
    init_pred = np.argmin(centroid_dists, axis=1)
    if detect_outliers:
        dists = np.min(centroid_dists, axis=1)
        mad = median_absolute_deviation(dists)
        outlier_mask = dists > outlier_factor * mad
        init_pred[outlier_mask] = 2
    return init_pred


def initialise_gmms(X, add_uniform=True, init_method='first-k',
                    batch_size=100, max_iter=4, pseudocount=0.5,
                    outlier_factor=0.5):
    '''
    Uses K-means to initialise 2-component GMM.
    Optionally adds a uniform dist to account for poorly aligned reads
    '''
    samp_sizes = [len(samp) for samp in X]
    X_pooled = np.concatenate(X)

    init_pred = kmeans_init_clusters(
        X_pooled, detect_outliers=add_uniform,
        init_method=init_method,
        batch_size=batch_size, max_iter=max_iter,
        outlier_factor=outlier_factor
    )
    dists = [
        pm.MultivariateGaussianDistribution.from_samples(X_pooled[init_pred == i])
        for i in range(N_COMPONENTS)
    ]

    ncomp = N_COMPONENTS + int(add_uniform)
    per_samp_preds = np.array_split(init_pred, np.cumsum(samp_sizes)[:-1])

    weights = [
        (np.bincount(pred, minlength=ncomp) + pseudocount) / (
            len(pred) + pseudocount * ncomp)
        for pred in per_samp_preds
    ]
    if add_uniform:
        outliers = X_pooled[init_pred == N_COMPONENTS]
        if len(outliers) == 0:
            # set params using all data
            outliers = X_pooled
        uniform_dist = pm.IndependentComponentsDistribution([
            pm.UniformDistribution(col.min(), col.max())
            for col in outliers.transpose()
        ])
        dists.append(uniform_dist)
    gmms = [
        pm.GeneralMixtureModel(dists, w) for w in weights
    ]
    return gmms, dists


def em(X, gmms, dists, max_iterations=1e8, stop_threshold=0.1,
       inertia=0.01, lr_decay=1e-3, pseudocount=0.5):
    '''Perform expectation maximisation'''
    initial_log_probability_sum = -np.inf
    iteration, improvement = 0, np.inf
    # perform EM
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
    return gmms, dists


def fit_multisamp_gmm(X, add_uniform=True, outlier_factor=0.5):
    '''
    Fit a gaussian mixture model to multiple samples. Each sample has its own
    GMM with its own weights but shares distributions with others. Returns
    a dists and weights both for combined data and for each sample
    '''
    try:
        with np.errstate(divide='ignore'):
            gmms, dists = initialise_gmms(X, add_uniform, outlier_factor=outlier_factor)

        with np.errstate(invalid='ignore', over='ignore'):
            gmms, dists = em(X, gmms, dists)

    except np.core._exceptions.UFuncTypeError:
        # sometimes small sample sizes cause fitting errors
        # actual error is caused by casting error for complex numbers
        # in pomegranate - see https://github.com/jmschrei/pomegranate/issues/633
        raise np.linalg.LinAlgError

    samp_sizes = [len(samp) for samp in X]
    per_samp_weights = np.array([np.exp(gmm.weights) for gmm in gmms])
    combined_weights = np.average(per_samp_weights, axis=0, weights=samp_sizes)
    return dists, combined_weights, per_samp_weights


def subsample(arr, max_size, random_state):
    '''If arr is longer than max_size, subsamples without replacement'''
    if len(arr) <= max_size:
        return arr
    else:
        idx = np.arange(len(arr))
        idx = random_state.choice(idx, size=max_size, replace=False)
        return arr[idx]


def orient_dists(dists, weights, per_samp_weights, centre):
    '''
    Orients distributions of a GMM so that the mus are in ascending
    order for the central kmer.
    '''
    assert 1 < len(dists) < 4
    has_uniform = len(dists) == 3
    lower_mean = dists[0].mu[centre]
    lower_std = np.sqrt(dists[0].cov[centre, centre])
    upper_mean = dists[1].mu[centre]
    upper_std = np.sqrt(dists[1].cov[centre, centre])
    if lower_mean > upper_mean:
        # flip model distributions and weights
        if has_uniform:
            dists = [dists[1], dists[0], dists[2]]
            weights = weights[[1, 0, 2]]
            per_samp_weights = per_samp_weights[:, [1, 0, 2]]
        else:
            dists = dists[::-1]
            weights = weights[::-1]
            per_samp_weights = per_samp_weights[:, ::-1]
        lower_mean, upper_mean = upper_mean, lower_mean
        lower_std, upper_std = upper_std, lower_std
    oriented_params = (
        lower_mean, lower_std, upper_mean, upper_std
    )
    return dists, weights, per_samp_weights, oriented_params


def fit_gmm(cntrl, treat, centre, add_uniform=True,
            outlier_factor=0.5, max_fit_depth=1000,
            random_state=None):
    '''
    Fits a multivariate, two-gaussian GMM to data and measures KL
    divergence of the resulting distributions.
    '''
    n_cntrl = len(cntrl)
    samp_sizes = np.array([len(samp) for samp in cntrl + treat])
    pooled = [
        subsample(samp, max_fit_depth, random_state) for samp in cntrl + treat
    ]
    dists, weights, per_samp_weights = fit_multisamp_gmm(
        pooled, add_uniform=add_uniform, outlier_factor=outlier_factor
    )

    dists, weights, per_samp_weights, params = orient_dists(
        dists, weights, per_samp_weights, centre
    )
    
    # use weights to estimate per-sample mod rates
    preds = np.round(per_samp_weights * samp_sizes[:, np.newaxis])
    cntrl_preds, treat_preds = preds[:n_cntrl], preds[n_cntrl:]
    gmm = pm.GeneralMixtureModel(dists, weights)
    return gmm, cntrl_preds, treat_preds, *params


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
    with np.errstate(invalid='raise'):
        try:
            het_g, p_val = two_cond_g_test([
                cntrl_preds[:, :2].sum(0), treat_preds[:, :2].sum(0)
            ])
        except FloatingPointError:
            # all classified as outliers!
            het_g = np.nan
            p_val = 1
    if p_val < p_val_threshold:
        cntrl_hom_g, _, = two_cond_g_test(cntrl_preds)
        treat_hom_g, _, = two_cond_g_test(treat_preds)
        hom_g = cntrl_hom_g + treat_hom_g
        if hom_g >= het_g:
            p_val = 1
    else:
        hom_g = np.nan
    return het_g, hom_g, p_val


def calculate_fractional_stats(cntrl_preds, treat_preds, pseudocount=0.5):
    '''
    Returns the relative modification rates (ignoring outliers) for treat
    and cntrl samples. Also calculates log ratio of mod:unmod reads.
    '''
    cntrl_pred = cntrl_preds.sum(0)
    treat_pred = treat_preds.sum(0)
    cntrl_frac_upper = (
        (cntrl_pred[1] + pseudocount) / 
        (cntrl_pred[:N_COMPONENTS].sum() + pseudocount)
    )
    treat_frac_upper = (
        (treat_pred[1] + pseudocount) / 
        (treat_pred[:N_COMPONENTS].sum() + pseudocount)
    )
    # symmetric log odds ratio
    log_odds = (np.log2(treat_frac_upper) - (1 - np.log2(treat_frac_upper))) - \
               (np.log2(cntrl_frac_upper) - (1 - np.log2(cntrl_frac_upper)))
    return cntrl_frac_upper, treat_frac_upper, log_odds


def format_sm_preds(sm_preds, sm_outlier, events):
    '''Create a json serialisable dict of single molecule predictions'''
    reps = events.index.get_level_values('replicate').tolist()
    read_ids = events.index.get_level_values('read_idx').tolist()
    events = events.values.tolist()
    sm_preds = sm_preds.tolist()
    sm_outlier = sm_outlier.tolist()
    sm_preds_dict = {}
    for r_id, rep, ev, p, o in zip(read_ids, reps, events, sm_preds, sm_outlier):
        try:
            sm_preds_dict[rep]['read_ids'].append(r_id)
        except KeyError:
            sm_preds_dict[rep] = {
                'read_ids': [r_id,],
                'events': [],
                'preds': [],
                'outlier_preds': [],
            }
        sm_preds_dict[rep]['events'].append(ev)
        sm_preds_dict[rep]['preds'].append(p)
        sm_preds_dict[rep]['outlier_preds'].append(o)
    return sm_preds_dict


def format_model(gmm):
    '''Create a json serialisable dict of GMM parameters'''
    model_json = {
        'unmod': {
            'mu': gmm.distributions[0].mu.tolist(),
            'cov': gmm.distributions[0].cov.tolist(),
        },
        'mod': {
            'mu': gmm.distributions[1].mu.tolist(),
            'cov': gmm.distributions[1].cov.tolist(),
        },
    }
    if len(gmm.distributions) == (N_COMPONENTS + 1):
        outlier = gmm.distributions[2]
        model_json['outlier'] = {
            'bounds': [u.parameters for u in outlier.distributions]
        }
    model_json['weights'] =  np.exp(gmm.weights).tolist()
    return model_json
    


@dataclasses.dataclass
class GMMTestResults:
    '''Class for handling results of positional_stats'''
    kmer: str = None
    log_odds: float = np.nan
    p_val: float = 1.
    fdr: float = 1.
    cntrl_frac_upper: float = np.nan
    treat_frac_upper: float = np.nan
    g_stat: float = np.nan
    hom_g_stat: float = np.nan
    lower_mean: float = np.nan
    lower_std: float = np.nan
    upper_mean: float = np.nan
    upper_std: float = np.nan
    ks_stat: float = np.nan
    current_shift_dir: str = None


def position_stats(cntrl, treat, kmers,
                   opts, random_state=None):
    '''
    Fits the GMM, estimates mod rates/changes, and performs G test
    '''
    window_size = len(kmers)
    centre = window_size // 2
    if not opts.model_dwell_time:
        # remove dwell time info
        cntrl = cntrl.iloc[:, :window_size]
        treat = treat.iloc[:, :window_size]
    kmer = kmers[centre]
    r = GMMTestResults(kmer=kmer)
    # first test that there is actually some difference in cntrl/treat
    r.ks_stat, ks_p_val = pca_kstest(cntrl.values, treat.values)

    # if there is we can perform the GMM fit and subsequent G test
    if r.ks_stat >= opts.min_ks and ks_p_val < opts.fdr_threshold:
        cntrl_fit_data = [c.values for _, c in cntrl.groupby('replicate', sort=False)]
        treat_fit_data = [t.values for _, t in treat.groupby('replicate', sort=False)]
        try:
            gmm, cntrl_preds, treat_preds, *fit_params = fit_gmm(
                cntrl_fit_data, treat_fit_data,
                centre=centre,
                add_uniform=opts.add_uniform,
                outlier_factor=opts.outlier_factor,
                max_fit_depth=opts.max_fit_depth,
                random_state=random_state,
            )
        except np.linalg.LinAlgError:
            return False, r, None
        std = max(
            np.sqrt(np.diagonal(gmm.distributions[0].cov).max()),
            np.sqrt(np.diagonal(gmm.distributions[1].cov).max()),
        )
        if std > opts.max_std:
            return False, r, None
 
        r.lower_mean, r.lower_std, r.upper_mean, r.upper_std = fit_params
        r.g_stat, r.hom_g_stat, r.p_val = gmm_g_test(
            cntrl_preds, treat_preds,
            p_val_threshold=opts.fdr_threshold
        )
        r.cntrl_frac_upper, r.treat_frac_upper, r.log_odds = calculate_fractional_stats(
            cntrl_preds, treat_preds
        )
        if r.p_val < opts.fdr_threshold and opts.generate_sm_preds:
            cntrl_prob = gmm.predict_proba(np.concatenate(cntrl_fit_data))[:, 1:].T
            treat_prob = gmm.predict_proba(np.concatenate(treat_fit_data))[:, 1:].T
            if opts.add_uniform:
                cntrl_prob, cntrl_outlier = cntrl_prob
                treat_prob, treat_outlier = treat_prob
            else:
                cntrl_outlier = np.zeros_like(cntrl_prob)
                treat_outlier = np.zeros_like(treat_prob)
            sm_preds = {
                'kmers': kmers.tolist(),
                'model': format_model(gmm),
                'cntrl': format_sm_preds(cntrl_prob, cntrl_outlier, cntrl),
                'treat': format_sm_preds(treat_prob, treat_outlier, treat),
            }
        else:
            sm_preds = None
        return True, r, sm_preds
    else:
        return False, r, None


def median_absolute_deviation_score(exp, obs):
    '''Calculates the MAD score of an array compared to an expected value'''
    return np.median(np.abs(obs - exp))


def assign_modified_distribution(results, sm_preds,
                                 model=load_model_priors(),
                                 sample_size=1000):
    '''
    Given all significant kmer results, predict for each kmer which distribution
    (i.e. upper or lower) is most likely to be modified using an existing model
    of nanopore current.    
    '''
    # corner case with no sig results
    if not len(results):
        return results, sm_preds
    res_assigned = []
    for kmer, kmer_res in results.groupby('kmer'):
        exp_mean = model.loc['level_mean', kmer]
        lower_fit_dist = median_absolute_deviation_score(kmer_res.lower_mean.values, exp_mean)
        upper_fit_dist = median_absolute_deviation_score(kmer_res.upper_mean.values, exp_mean)
        if lower_fit_dist <= upper_fit_dist:
            # lower is unmod
            kmer_res['current_shift_dir'] = 'h'
        else:
            # higher is unmod, flip the values
            kmer_res.loc[:, 'current_shift_dir'] = 'l'
            kmer_res.loc[:, 'cntrl_frac_upper'] = 1 - kmer_res.loc[:, 'cntrl_frac_upper']
            kmer_res.loc[:, 'treat_frac_upper'] = 1 - kmer_res.loc[:, 'treat_frac_upper']
            kmer_res.loc[:, 'log_odds'] = np.negative(kmer_res.loc[:, 'log_odds'])
            kmer_res.loc[:, ['lower_mean','upper_mean']] = (
                kmer_res.loc[:, ['upper_mean','lower_mean']].values
            )
            kmer_res.loc[:, ['lower_std','upper_std']] = (
                kmer_res.loc[:, ['upper_std','lower_std']].values
            )
            # also need to reverse the sm_preds:
            for _, gene_id, pos in kmer_res[['gene_id', 'pos']].itertuples():
                try:
                    pos_sm_preds = sm_preds[gene_id][pos]
                except KeyError:
                    continue
                m = pos_sm_preds['model']
                m['unmod'], m['mod'] = m['mod'], m['unmod']
                for cond in ['cntrl', 'treat']:
                    for rep_sm_preds in pos_sm_preds[cond].values():
                        rep_sm_preds['preds'] = [
                            1 - (p + o) for p, o in zip(rep_sm_preds['preds'],
                                                        rep_sm_preds['outlier_preds'])
                        ]
        res_assigned.append(kmer_res)
    results = pd.concat(res_assigned)
    return results, sm_preds