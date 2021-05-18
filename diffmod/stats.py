import dataclasses

import numpy as np
from scipy import stats
import pomegranate as pm

from .io import load_model_priors


def get_random_projections(n_projections, d, seed=None):
    """
    Generates n_projections samples from the uniform on the unit sphere of dimension d-1`
    """

    if not isinstance(seed, np.random.RandomState):
        random_state = np.random.RandomState(seed)
    else:
        random_state = seed

    projections = random_state.normal(0., 1., [n_projections, d])
    norm = np.linalg.norm(projections, ord=2, axis=1, keepdims=True)
    projections = projections / norm
    return projections


def sliced_earth_mover_distance(dist1, dist2, sample_size=1000,
                                n_projections=50, seed=None):
    """
    Computes a Monte-Carlo approximation of the 2-Sliced Wasserstein distance
    modified from python optimal transport (POT)
    """

    X1 = dist1.sample(sample_size)
    X2 = dist2.sample(sample_size)
    n_dim = X1.shape[1]

    projections = get_random_projections(n_projections, n_dim, seed)

    X1_projections = np.dot(projections, X1.T)
    X2_projections = np.dot(projections, X2.T)

    res = 0.

    for i, (X1_proj, X2_proj) in enumerate(zip(X1_projections, X2_projections)):
        emd = stats.wasserstein_distance(X1_proj, X2_proj)
        res += emd

    res = (res / n_projections) ** 0.5
    return res


def fit_multisamp_gmm(X, n_components, init, max_iterations=1e8, stop_threshold=0.1,
                      inertia=0.01, lr_decay=1e-3, pseudocount=0.5):
    '''
    Fit a gaussian mixture model to multiple samples. Each sample has its own
    GMM with its own weights but shares distributions with others. Returns
    a dists and weights both for combined data and for each sample
    '''
    samp_sizes = [len(samp) for samp in X]
    X_pooled = np.concatenate(X)
    n_dim = X_pooled.shape[1]
    dists = [
        pm.IndependentComponentsDistribution([
            pm.NormalDistribution(*i) for i in comp_init])
        for comp_init in init   
    ]
    # dists are shared, weights are not
    gmms = [pm.GeneralMixtureModel(dists) for _ in X]
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

    per_samp_weights = np.array([np.exp(gmm.weights) for gmm in gmms])
    combined_weights = np.average(per_samp_weights, axis=0, weights=samp_sizes)
    return dists, combined_weights, per_samp_weights


def generate_init_points(expected_params, n_components):
    '''
    Heuristic to generate starting points for fitting GMM - faster than Kmeans
    '''
    expected = pm.IndependentComponentsDistribution([
        pm.NormalDistribution(*e) for e in expected_params
    ])
    init = [
        [[mu, sig] for mu, (_, sig) in zip(sp, expected_params)]
        for sp in expected.sample(n_components)
    ]
    return expected, init


def subsample(arr, max_size):
    if len(arr) <= max_size:
        return arr
    else:
        idx = np.arange(len(arr))
        idx = np.random.choice(idx, size=max_size, replace=False)
        return arr[idx]


def fit_gmm(cntrl, treat, expected_params,
            max_fit_depth=1000, min_mod_vs_unmod_emd=0.5):
    '''
    Fits a multivariate, two-gaussian GMM to data and measures KL
    divergence of the resulting distributions.
    '''
    n_cntrl = len(cntrl)
    pooled = cntrl + treat
    samp_sizes = np.array([len(samp) for samp in pooled])
    pooled = [
        subsample(samp, max_fit_depth) for samp in pooled
    ]
    expected, init = generate_init_points(expected_params, 2)
    dists, weights, per_samp_weights = fit_multisamp_gmm(pooled, n_components=2, init=init)
    # use weights to estimate per-sample mod rates
    preds = np.round(per_samp_weights * samp_sizes[:, np.newaxis])
    emd = sliced_earth_mover_distance(dists[0], dists[1])

    if emd >= min_mod_vs_unmod_emd:
        # assess which dist is closer to expectation:
        dist_from_expected = [sliced_earth_mover_distance(expected, icd) for icd in dists]
        sort_idx = np.argsort(dist_from_expected)
        expected_emd = min(dist_from_expected)
        # sort so that null model is zero
        dists = [dists[i] for i in sort_idx]
        weights = weights[sort_idx]
        preds = preds[:, sort_idx]
    else:
        # no need to bother with expensive EMDs
        expected_emd = np.inf

    gmm = pm.GeneralMixtureModel(dists, weights)
    cntrl_preds, treat_preds = preds[:n_cntrl], preds[n_cntrl:]
    return gmm, emd, expected_emd, cntrl_preds, treat_preds


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


def get_current_shift_direction(gmm, centre):
    unmod_dist, mod_dist = gmm.distributions
    unmod_mean, unmod_std = unmod_dist.distributions[centre].parameters
    mod_mean, mod_std = mod_dist.distributions[centre].parameters
    shift_dir = 'h' if mod_mean > unmod_mean else 'l'
    return mod_mean, mod_std, unmod_mean, unmod_std, shift_dir


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


@dataclasses.dataclass
class GMMTestResults:
    '''Class for handling results of positional_stats'''
    kmer: str = None
    log_odds: float = np.nan
    p_val: float = np.nan
    fdr: float = 1.
    cntrl_frac_mod: float = np.nan
    treat_frac_mod: float = np.nan
    g_stat: float = np.nan
    hom_g_stat: float = np.nan
    mod_mean: float = np.nan
    mod_std: float = np.nan
    unmod_mean: float = np.nan
    umod_std: float = np.nan
    mod_dir: str = None
    emd: float = np.nan
    ks_stat: float = np.nan
    ks_p_val: float = np.nan


def position_stats(cntrl, treat, kmers,
                   max_fit_depth=1000,
                   max_cntrl_vs_exp_emd=1,
                   min_mod_vs_unmod_emd=0.5,
                   min_ks=0.05, p_val_threshold=0.05,
                   model=load_model_priors(),
                   generate_sm_preds=False):
    '''
    Fits the GMM, estimates mod rates/changes, and performs G test
    '''
    window_size = len(kmers)
    centre = window_size // 2
    kmer = kmers[centre]
    r = GMMTestResults(kmer=kmer)
    # first test that there is actually some difference in cntrl/treat
    # easiest way to do this is to just test the central kmer...
    pass_kstest = False
    pass_emd = False
    r.ks_stat, r.ks_p_val = stats.ks_2samp(
        cntrl['mean'].values[:, centre],
        treat['mean'].values[:, centre],
    )
    # if there is we can perform the GMM fit and subsequent G test
    if r.ks_stat >= min_ks and r.ks_p_val < p_val_threshold:
        pass_kstest = True
        expected_params = model.loc[:, kmers]
        expected_params = expected_params.values.reshape(2, -1).T
        cntrl_fit_data = [c.values for _, c in cntrl.groupby('replicate')]
        treat_fit_data = [t.values for _, t in treat.groupby('replicate')]
        gmm, r.emd, exp_emd, cntrl_preds, treat_preds = fit_gmm(
            cntrl_fit_data, treat_fit_data, expected_params,
            max_fit_depth, min_mod_vs_unmod_emd
        )
        # if the KL divergence of the distributions is too small we stop here
        if r.emd >= min_mod_vs_unmod_emd and exp_emd <= max_cntrl_vs_exp_emd and r.emd > exp_emd:
            pass_emd = True
            r.g_stat, r.hom_g_stat, r.p_val = gmm_g_test(
                cntrl_preds, treat_preds,
                p_val_threshold=p_val_threshold
            )
    
    if pass_emd & pass_kstest:
        # sort out the single molecule predictions
        r.mod_mean, r.mod_std, r.unmod_mean, r.umod_std, r.mod_dir = get_current_shift_direction(gmm, centre) 
        r.cntrl_frac_mod, r.treat_frac_mod, r.log_odds = calculate_mod_stats(
            cntrl_preds, treat_preds
        )
        if r.p_val < p_val_threshold and generate_sm_preds:
            cntrl_probs = gmm.predict_proba(np.concatenate(cntrl_fit_data))[:, 1]
            treat_probs = gmm.predict_proba(np.concatenate(treat_fit_data))[:, 1]
            sm_preds = {
                'kmers' : kmers.tolist(),
                'cntrl' : format_sm_preds(cntrl_probs, cntrl),
                'treat' : format_sm_preds(treat_probs, treat),
            }
        else:
            sm_preds = None
        return True, r, sm_preds
    else:
        return False, r, None