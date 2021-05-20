import dataclasses

import numpy as np
import pandas as pd
from scipy import stats
import pomegranate as pm

from .io import load_model_priors


def fit_multisamp_gmm(X, max_iterations=1e8, stop_threshold=0.1,
                      inertia=0.01, lr_decay=1e-3, pseudocount=0.5):
    '''
    Fit a gaussian mixture model to multiple samples. Each sample has its own
    GMM with its own weights but shares distributions with others. Returns
    a dists and weights both for combined data and for each sample
    '''
    samp_sizes = [len(samp) for samp in X]

    X_pooled = np.concatenate(X)
    n_dim = X_pooled.shape[1]
    kmeans = pm.Kmeans(2, init='first-k')
    kmeans.fit(X_pooled, batch_size=100, max_iterations=4)
    init_pred = kmeans.predict(X_pooled)
    dists = [
        pm.MultivariateGaussianDistribution.from_samples(X_pooled[init_pred == i])
        for i in range(2)
    ]
    weights = [
        np.mean(pred) for pred in
        np.array_split(init_pred, np.cumsum(samp_sizes)[:-1])
    ]
    # dists are shared, weights are not
    gmms = [pm.GeneralMixtureModel(dists, [1 - w, w]) for w in weights]
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


def subsample(arr, max_size, random_state):
    if len(arr) <= max_size:
        return arr
    else:
        idx = np.arange(len(arr))
        idx = random_state.choice(idx, size=max_size, replace=False)
        return arr[idx]


def fit_gmm(cntrl, treat, centre, max_fit_depth=1000, random_state=None):
    '''
    Fits a multivariate, two-gaussian GMM to data and measures KL
    divergence of the resulting distributions.
    '''
    n_cntrl = len(cntrl)
    samp_sizes = np.array([len(samp) for samp in cntrl + treat])
    pooled = [
        subsample(samp, max_fit_depth, random_state) for samp in cntrl + treat
    ]
    try:
        dists, weights, per_samp_weights = fit_multisamp_gmm(pooled)
    except np.core._exceptions.UFuncTypeError:
        raise np.linalg.LinAlgError

    lower_mean = dists[0].parameters[0][centre]
    lower_std = np.sqrt(dists[0].parameters[1][centre][centre])
    upper_mean = dists[1].parameters[0][centre]
    upper_std = np.sqrt(dists[1].parameters[1][centre][centre])

    if lower_mean > upper_mean:
        # flip model distributions and weights
        dists = dists[::-1]
        weights = weights[::-1]
        per_samp_weights = per_samp_weights[:, ::-1]
        lower_mean, upper_mean = upper_mean, lower_mean
        lower_std, upper_std = upper_std, lower_std
    
    # use weights to estimate per-sample mod rates
    preds = np.round(per_samp_weights * samp_sizes[:, np.newaxis])
    gmm = pm.GeneralMixtureModel(dists, weights)
    cntrl_preds, treat_preds = preds[:n_cntrl], preds[n_cntrl:]
    return (gmm, cntrl_preds, treat_preds,
            lower_mean, lower_std, upper_mean, upper_std)


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


def calculate_fractional_stats(cntrl_preds, treat_preds, pseudocount=0.5):
    cntrl_pred = cntrl_preds.sum(0)
    treat_pred = treat_preds.sum(0)
    cntrl_frac_upper = (cntrl_pred[1] + pseudocount) / (cntrl_pred.sum() + pseudocount)
    treat_frac_upper = (treat_pred[1] + pseudocount) / (treat_pred.sum() + pseudocount)
    log_odds = np.log2(treat_frac_upper) - np.log2(cntrl_frac_upper)
    return cntrl_frac_upper, treat_frac_upper, log_odds


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
                   max_fit_depth=1000,
                   min_ks=0.1, p_val_threshold=0.05,
                   model_dwell_time=False,
                   generate_sm_preds=False,
                   random_state=None):
    '''
    Fits the GMM, estimates mod rates/changes, and performs G test
    '''
    window_size = len(kmers)
    centre = window_size // 2
    if not model_dwell_time:
        # remove dwell time info
        cntrl = cntrl.iloc[:, :window_size]
        treat = treat.iloc[:, :window_size]
    kmer = kmers[centre]
    r = GMMTestResults(kmer=kmer)
    # first test that there is actually some difference in cntrl/treat
    # easiest way to do this is to just test the central kmer...
    r.ks_stat, ks_p_val = stats.ks_2samp(
        cntrl['mean'].values[:, centre],
        treat['mean'].values[:, centre],
    )
    # if there is we can perform the GMM fit and subsequent G test
    if r.ks_stat >= min_ks and ks_p_val < p_val_threshold:
        cntrl_fit_data = [c.values for _, c in cntrl.groupby('replicate')]
        treat_fit_data = [t.values for _, t in treat.groupby('replicate')]
        try:
            gmm, cntrl_preds, treat_preds, *fit_params = fit_gmm(
                cntrl_fit_data, treat_fit_data,
                centre, max_fit_depth, random_state,
            )
        except np.linalg.LinAlgError:
            return False, r, None
        r.lower_mean, r.lower_std, r.upper_mean, r.upper_std = fit_params
        r.g_stat, r.hom_g_stat, r.p_val = gmm_g_test(
            cntrl_preds, treat_preds,
            p_val_threshold=p_val_threshold
        )
        r.cntrl_frac_upper, r.treat_frac_upper, r.log_odds = calculate_fractional_stats(
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


def median_abs_deviation(exp, obs):
    return np.median(np.abs(obs - exp))


def assign_modified_distribution(results, sm_preds,
                                 model=load_model_priors(),
                                 sample_size=1000):
    # corner case with no sig results
    if not len(results):
        return results, sm_preds
    res_assigned = []
    for kmer, kmer_res in results.groupby('kmer'):
        exp_mean = model.loc['current_mean', kmer]
        lower_fit_dist = median_abs_deviation(kmer_res.lower_mean.values, exp_mean)
        upper_fit_dist = median_abs_deviation(kmer_res.upper_mean.values, exp_mean)
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
                for cond in ['cntrl', 'treat']:
                    for rep_sm_preds in pos_sm_preds[cond].values():
                        rep_sm_preds['preds'] = [
                            1 - p for p in rep_sm_preds['preds']
                        ]
        res_assigned.append(kmer_res)
    results = pd.concat(res_assigned)
    return results, sm_preds