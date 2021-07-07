from collections import defaultdict
import dataclasses

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import Table2x2
import pomegranate as pm

from .io import load_model_priors


N_COMPONENTS = 2


def pca_comp1(X, standardise=True):
    '''Quick principal components with 1 comp'''
    if standardise:
        mu = X.mean(axis=0)
        sig = np.sqrt(((X - mu) ** 2.0).mean(0))
        X = (X - mu) / sig
    u, s, v = np.linalg.svd(X, full_matrices=False)
    vecs = v.T
    i1 = np.argmax(s ** 2.0)
    vecs = vecs[:, i1, np.newaxis]
    comp1 = X.dot(vecs)
    return comp1.ravel()


def _ks_d(data1, data2, pooled):
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    cdf1 = np.searchsorted(data1, pooled, side='right')/(1.0 * n1)
    cdf2 = (np.searchsorted(data2, pooled, side='right'))/(1.0 * n2)
    d = np.max(np.absolute(cdf1 - cdf2))
    en = np.sqrt(n1 * n2 / float(n1 + n2))
    return d, en


def ks_2samp(data1, data2, pooled):
    d, en = _ks_d(data1, data2, pooled)
    prob = stats.kstwobign.sf((en + 0.12 + 0.11 / en) * d)
    return d, prob


def subsample(arr, max_size, random_state):
    '''If arr is longer than max_size, subsamples without replacement'''
    if len(arr) <= max_size:
        return arr
    else:
        idx = np.arange(len(arr))
        idx = random_state.choice(idx, size=max_size, replace=False)
        return arr[idx]


def pca_kstest(cntrl_data, treat_data, max_size, random_state):
    '''
    Transform multivariate data to univariate using PCA and perform
    Kolmogorov-Smirnov test
    '''
    cntrl_data = subsample(cntrl_data, max_size, random_state)
    treat_data = subsample(treat_data, max_size, random_state)
    n_cntrl = len(cntrl_data)
    pooled = np.concatenate([cntrl_data, treat_data])
    comps = pca_comp1(pooled)
    ks, p_val = ks_2samp(comps[:n_cntrl], comps[n_cntrl:], comps)
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
        init_pred[outlier_mask] = N_COMPONENTS
    return init_pred


def initialise_gmms(X, covariance='full', add_uniform=True,
                    init_method='first-k',
                    batch_size=100, max_iter=4, pseudocount=0.5,
                    outlier_factor=0.5):
    '''
    Uses K-means to initialise 2-component GMM.
    Optionally adds a uniform dist to account for poorly aligned reads
    '''
    samp_sizes = [len(samp) for samp in X]
    X_pooled = np.concatenate(X)
    n_dim = X_pooled.shape[1]

    init_pred = kmeans_init_clusters(
        X_pooled, detect_outliers=add_uniform,
        init_method=init_method,
        batch_size=batch_size, max_iter=max_iter,
        outlier_factor=outlier_factor
    )
    if covariance == 'full':
        dists = [
            pm.MultivariateGaussianDistribution.from_samples(X_pooled[init_pred == i])
            for i in range(N_COMPONENTS)
        ]
    elif covariance == 'diag':
        dists = []
        for i in range(N_COMPONENTS):
            X_i = X_pooled[init_pred == i]
            if len(X_i) == 0:
                dists.append(pm.IndependentComponentsDistribution([
                    pm.NormalDistribution(0, 1)
                    for j in range(n_dim)
                ]))
            else:
                dists.append(pm.IndependentComponentsDistribution([
                    pm.NormalDistribution(X_i[:, j].mean(), X_i[:, j].std())
                    for j in range(n_dim)
                ]))
    else:
        raise ValueError('Only full and diag covariances are supported')

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


def fit_multisamp_gmm(X, covariance='full', add_uniform=True, outlier_factor=0.5):
    '''
    Fit a gaussian mixture model to multiple samples. Each sample has its own
    GMM with its own weights but shares distributions with others. Returns
    a dists and weights both for combined data and for each sample
    '''
    try:
        with np.errstate(divide='ignore'):
            gmms, dists = initialise_gmms(X, covariance, add_uniform, outlier_factor=outlier_factor)

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


def fit_gmm(cntrl, treat, covariance='full', add_uniform=True,
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
        pooled, covariance=covariance,
        add_uniform=add_uniform, outlier_factor=outlier_factor
    )

    if covariance == 'full':
        model_params = (
            dists[0].mu, dists[1].mu,
            np.sqrt(np.diagonal(dists[0].cov)), np.sqrt(np.diagonal(dists[1].cov))
        )
    elif covariance == 'diag':
        mu_1, std_1 = zip(
            *[d.parameters for d in dists[0]]
        )
        mu_2, std_2 = zip(
            *[d.parameters for d in dists[1]]
        )
        model_params = [
            np.array(mu_1), np.array(mu_2),
            np.array(std_1), np.array(std_2)
        ]
    else:
        raise ValueError('Only full and diag covariances are supported')
    
    # use weights to estimate per-sample mod rates
    preds = np.round(per_samp_weights * samp_sizes[:, np.newaxis])
    cntrl_preds, treat_preds = preds[:n_cntrl], preds[n_cntrl:]
    gmm = pm.GeneralMixtureModel(dists, weights)
    return gmm, cntrl_preds, treat_preds, model_params


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


def calculate_fractional_stats(cntrl_preds, treat_preds, pseudocount=0.5, ci=95):
    '''
    Returns the relative modification rates (ignoring outliers) for treat
    and cntrl samples. Also calculates log ratio of mod:unmod reads.
    '''
    cntrl_pred = cntrl_preds.sum(0)
    treat_pred = treat_preds.sum(0)
    with np.errstate(invalid='ignore'):
        cntrl_frac_upper = cntrl_pred[1] / cntrl_pred[:N_COMPONENTS].sum()
        treat_frac_upper = treat_pred[1] / treat_pred[:N_COMPONENTS].sum()

    ct = Table2x2([cntrl_pred[:N_COMPONENTS], treat_pred[:N_COMPONENTS]], shift_zeros=True)
    log_odds = ct.log_oddsratio
    log_odds_ci = ct.log_oddsratio_confint(alpha=1 - ci / 100)
    return cntrl_frac_upper, treat_frac_upper, log_odds, *log_odds_ci


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
    try:
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
    except AttributeError:
        # model is IndependentComponentsDistribution (diagonal covariance)
        mu_1, std_1 = zip(
            *[d.parameters for d in gmm.distributions[0]]
        )
        cov_1 = np.diag(std_1 ** 2)
        mu_2, std_2 = zip(
            *[d.parameters for d in gmm.distributions[1]]
        )
        cov_2 = np.diag(std_2 ** 2)
        model_json = {
            'unmod': {
                'mu': mu_1, 'cov': cov_1,
            },
            'mod': {
                'mu': mu_2, 'cov': cov_2,
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
    kmer: str
    kmers: np.ndarray
    centre: int
    log_odds: float = np.nan
    log_odds_lower_ci: float = np.nan
    log_odds_upper_ci: float = np.nan
    p_val: float = 1.
    fdr: float = 1.
    cntrl_frac_mod: float = np.nan
    treat_frac_mod: float = np.nan
    g_stat: float = np.nan
    hom_g_stat: float = np.nan
    dist1_means: np.ndarray = None
    dist1_stds: np.ndarray = None
    dist2_means: np.ndarray = None
    dist2_stds: np.ndarray = None
    ks_stat: float = np.nan


def position_stats(cntrl, treat, kmers,
                   opts, random_state=None):
    '''
    Fits the GMM, estimates mod rates/changes, and performs G test
    '''
    window_size = len(kmers)
    centre = window_size // 2
    kmer = kmers[centre]
    r = GMMTestResults(kmer=kmer, kmers=kmers, centre=centre)
    # first test that there is actually some difference in cntrl/treat
    r.ks_stat, ks_p_val = pca_kstest(
        cntrl.values, treat.values,
        max_size=opts.max_fit_depth,
        random_state=random_state,
    )
    if not opts.gmm: # no modelling
        r.p_val = ks_p_val
        return True, r, None

    # if there is we can perform the GMM fit and subsequent G test
    if r.ks_stat >= opts.min_ks and ks_p_val < opts.fdr_threshold:
        cntrl_fit_data = [c.values for _, c in cntrl.groupby('replicate', sort=False)]
        treat_fit_data = [t.values for _, t in treat.groupby('replicate', sort=False)]
        try:
            gmm, cntrl_preds, treat_preds, model_params = fit_gmm(
                cntrl_fit_data, treat_fit_data,
                centre=centre,
                covariance=opts.covariance_type,
                add_uniform=opts.add_uniform,
                outlier_factor=opts.outlier_factor,
                max_fit_depth=opts.max_fit_depth,
                random_state=random_state,
            )
        except np.linalg.LinAlgError:
            return False, r, None

        r.dist1_means, r.dist2_means, r.dist1_stds, r.dist2_stds = model_params
 
        r.g_stat, r.hom_g_stat, r.p_val = gmm_g_test(
            cntrl_preds, treat_preds,
            p_val_threshold=opts.fdr_threshold
        )

        frac_stats = calculate_fractional_stats(cntrl_preds, treat_preds)
        (
            r.cntrl_frac_mod, r.treat_frac_mod,
            r.log_odds, r.log_odds_lower_ci, r.log_odds_upper_ci
        ) = frac_stats

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


def calculate_kmer_shift_directions(results, expected_model):
    results = results[['kmers', 'dist1_means', 'dist2_means']]
    kmer_lower_dists = defaultdict(list)
    kmer_upper_dists = defaultdict(list)
    for _, kmers, dist1_fit_means, dist2_fit_means in results.itertuples():
        for k, lm, um in zip(kmers, dist1_fit_means, dist2_fit_means):
            if lm > um:
                lm, um = um, lm
            exp = expected_model.loc['level_mean', k]
            kmer_lower_dists[k].append(abs(lm - exp))
            kmer_upper_dists[k].append(abs(um - exp))

    kmer_shift_dirs = {}
    for k in kmer_lower_dists:
        if np.median(kmer_upper_dists[k]) > np.median(kmer_lower_dists[k]):
            kmer_shift_dirs[k] = 1 # upper is more likely modified 
        else:
            kmer_shift_dirs[k] = 0 # lower is more likely modified
    return kmer_shift_dirs


def dist1_is_modified(kmers, dist1_means, dist2_means, kmer_shift_dirs):
    '''
    Predict whether dist1 or dist2 is modified using global shift
    directions for each kmer in kmers and the separation between
    dist1 and dist2
    '''
    diff = dist1_means - dist2_means
    dist1_score = 0
    dist2_score = 0
    for k, d in zip(kmers, diff):
        upper_is_mod = kmer_shift_dirs[k]
        d2_is_upper = d < 0
        if upper_is_mod ^ d2_is_upper:
            dist1_score += abs(d)
        else:
            dist2_score += abs(d)
    # larger score is modified
    return dist1_score > dist2_score


def assign_modified_distribution(results, sm_preds,
                                 model=load_model_priors(),
                                 sample_size=1000):
    '''
    Given all significant kmer results, predict for each kmer which distribution
    is most likely to be modified using an existing model of nanopore current.    
    '''
    # corner case with no sig results
    if not len(results):
        return results, sm_preds
    kmer_shift_dirs = calculate_kmer_shift_directions(results, model)
    res_assigned = []
    for i in results.index:
        kmers, dist1_means, dist2_means = results.loc[i, ['kmers', 'dist1_means', 'dist2_means']]
        if dist1_is_modified(kmers, dist1_means, dist2_means, kmer_shift_dirs):
            # higher is unmod, flip the values
            results.loc[i, 'cntrl_frac_mod'] = 1 - results.loc[i, 'cntrl_frac_mod']
            results.loc[i, 'treat_frac_mod'] = 1 - results.loc[i, 'treat_frac_mod']
            results.loc[i, 'log_odds'] = np.negative(results.loc[i, 'log_odds'])
            results.loc[i, ['log_odds_lower_ci', 'log_odds_upper_ci']] = np.negative(
                results.loc[i, ['log_odds_upper_ci', 'log_odds_lower_ci']]
            )
            results.loc[i, ['dist1_means','dist2_means']] = (
                results.loc[i, ['dist2_means','dist1_means']].values
            )
            results.loc[i, ['dist1_stds','dist2_stds']] = (
                results.loc[i, ['dist2_stds','dist1_stds']].values
            )
            # also need to reverse the sm_preds:
            gene_id, pos = results.loc[i, ['gene_id', 'pos']]
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
    return results, sm_preds