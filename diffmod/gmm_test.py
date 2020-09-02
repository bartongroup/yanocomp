import os
from functools import partial
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import h5py as h5
import pomegranate as pm

import click


RESULTS_COLUMNS = [
    'chrom', 'pos', 'gene_id', 'strand',
    'log_odds', 'p_val', 'fdr',
    'cntrl_frac_mod', 'treat_frac_mod',
    'g_stat', 'hom_g_stat', 'kl_divergence',
]


def get_model(model_fn=None):
    if model_fn is None:
        model_fn = os.path.join(
            od.path.split(os.path.abspath(__file__))[0],
            'data/r9.4_70bps.u_to_t_rna.5mer.template.model'
        )
    m = pd.read_csv(
        model_fn, sep='\t', comment='#', index_col='kmer'
    )
    return m[['level_mean', 'level_stdv']].T.to_dict(orient='list')


class HDF5List:

    def __init__(self, hdf5_fns):
        self._hdf5_list = [
            h5.File(fn, 'r') for fn in hdf5_fns
        ]

    def close(self):
        for f in self._hdf5_list:
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def __iter__(self):
        return iter(self._hdf5_list)

    def __add__(self, other):
        return self._hdf5_list + other._hdf5_list


def get_expressed_genes(datasets):
    # filter out any transcripts that are not expressed in all samples
    genes = set(datasets[0].keys())
    for d in datasets[1:]:
        genes.intersection_update(set(d.keys()))
    return list(genes)


def fetch_gene_events(gene_id, datasets,
                      fetch_transcript_ids=False,
                      fetch_read_ids=False):
    gene_events = []
    for i, d in enumerate(datasets, 1):
        # read full dataset from disk
        e = pd.DataFrame(d[f'{gene_id}/events'][:])
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
        treat_depth = cntrl_pos_events.replicate.value_counts()
        if (cntrl_depth >= min_depth).all() and (treat_depth >= min_depth).all():
            yield chrom, pos, strand, kmer, cntrl_pos_events, treat_pos_events


def kl_divergence(X_mu, X_sigma, Y_mu, Y_sigma):
    X_var, Y_var = X_sigma ** 2, Y_sigma ** 2
    lsr = np.log(Y_sigma / X_sigma)
    sdm = (X_mu - Y_mu) ** 2
    return  lsr + (X_var + sdm) / (2 * Y_var) - 0.5


def fit_gmm(cntrl_means, treat_means, expected_params, max_gmm_fit_depth=1000):
    pooled = np.concatenate([cntrl_means, treat_means])
    if len(pooled) > max_gmm_fit_depth:
        np.random.shuffle(pooled)
        pooled = pooled[:max_gmm_fit_depth]
    # Model starts as two identical Normal distributions initialised from
    # the expected parameters. The first cannot be fit.
    gmm = pm.GeneralMixtureModel(
        [
            pm.NormalDistribution(*expected_params, frozen=True),
            pm.NormalDistribution(*expected_params),
        ]
    )
    gmm.fit(pooled.reshape(-1, 1))
    kld = kl_divergence(
        *gmm.distributions[1].parameters,
        *gmm.distributions[0].parameters,
    )
    return gmm, kld


def gmm_predictions(gmm, means):
    probs = gmm.predict_proba(means.reshape(-1, 1))
    preds = probs.argmax(1)
    log_prob = np.log2(probs.sum(0))
    log_ratio = log_prob[1] - log_prob[0]
    frac_mod = preds.sum() / len(preds)
    return preds, frac_mod, log_ratio


def crosstab_preds_per_replicate(preds, replicates):
    # make categorical to catch cases where 100% of preds are one class
    preds = pd.Categorical(preds, categories=[0, 1])
    ct = pd.crosstab(replicates, preds, dropna=False).values
    return ct


def two_cond_g_test(counts):
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
                   max_gmm_fit_depth=1000,
                   min_kld=0.5,
                   p_val_threshold=0.05,
                   model=get_model()):
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
            cntrl['mean'], treat['mean'], expected_params,
            max_gmm_fit_depth
        )
        fit_mean, fit_std = gmm.distributions[1].parameters
        # if the KL divergence of the distributions is too small we stop here
        if kld >= min_kld:
            pass_kld = True
            cntrl_preds, cntrl_frac_mod, cntrl_lr = gmm_predictions(
                gmm, cntrl['mean'].values
            )
            treat_preds, treat_frac_mod, treat_lr = gmm_predictions(
                gmm, treat['mean'].values
            )
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


def test_chunk(gene_ids, cntrl_fns, treat_fns,
               min_depth=10, max_gmm_fit_depth=1000,
               min_kld=0.5, p_val_threshold=0.05):
    chunk_res = []
    with HDF5List(cntrl_fns) as cntrl_h5, HDF5List(treat_fns) as treat_h5:
        for gene_id in gene_ids:
            pos_iter = iter_positions(gene_id, cntrl_h5, treat_h5, min_depth)
            for chrom, pos, strand, kmer, cntrl, treat in pos_iter:
                r = position_stats(
                    cntrl, treat, kmer,
                    max_gmm_fit_depth,
                    min_kld, p_val_threshold
                )
                if r[0]:
                    r = [chrom, pos, gene_id, strand] + r[1:]
                    chunk_res.append(r)
    return chunk_res


def parallel_test(cntrl_fns, treat_fns,
                  min_depth=10, max_gmm_fit_depth=1000,
                  min_kld=0.5, p_val_threshold=0.05,
                  processes=1):

    with HDF5List(cntrl_fns) as cntrl_h5, HDF5List(treat_fns) as treat_h5:
        gene_ids = get_expressed_genes(cntrl_h5 + treat_h5)
        gene_id_chunks = np.array_split(gene_ids, processes)

    with mp.Pool(processes) as pool:
        _test_chunk = partial(
            test_chunk,
            cntrl_fns=cntrl_fns,
            treat_fns=treat_fns,
            min_depth=min_depth,
            max_gmm_fit_depth=max_gmm_fit_depth,
            min_kld=min_kld,
            p_val_threshold=p_val_threshold
        )
        res = []
        for chunk_res in pool.imap_unordered(_test_chunk, gene_id_chunks):
            res += chunk_res

    res = pd.DataFrame(res, columns=RESULTS_COLUMNS)
    _, res['fdr'], _, _ = multipletests(res.p_val, method='fdr_bh')
    return res


def to_bed(res, output_bed_fn, fdr_threshold=0.05):
    sig_res = res.query(f'fdr < {fdr_threshold}')
    sig_res = sig_res.sort_values(by=['chrom', 'pos'])
    with open(output_bed_fn, 'w') as bed:
        for record in sig_res.itertuples(index=False):
            (chrom, pos, gene_id, strand,
             log_odds, pval, fdr, c_fm, t_fm,
             g_stat, hom_g_stat, kld) = record
            nlogfdr = - int(round(np.log10(fdr)))
            bed_record = (
                f'{chrom:s}\t{pos - 2:d}\t{pos + 3:d}\t'
                f'{gene_id}\t{nlogfdr:d}\t{strand:s}\t'
                f'{log_odds:.2f}\t{pval:.2g}\t{fdr:.2g}\t'
                f'{c_fm:.2f}\t{t_fm:.2f}\t'
                f'{g_stat:.2f}\t{hom_g_stat:.2f}\t{kld:.2f}\n'
            )
            bed.write(bed_record)


@click.command()
@click.option('-c', '--cntrl-hdf5-fns', required=True, multiple=True)
@click.option('-t', '--treat-hdf5-fns', required=True, multiple=True)
@click.option('-b', '--output-bed-fn', required=True)
@click.option('-n', '--min-depth', required=False, default=10)
@click.option('-m', '--max-gmm-fit-depth', required=False, default=1000)
@click.option('-k', '--min-kl-divergence', required=False, default=0.5)
@click.option('-f', '--fdr-threshold', required=False, default=0.05)
@click.option('-p', '--processes', required=False, default=1)
def gmm_test(cntrl_hdf5_fns, treat_hdf5_fns, output_bed_fn,
             min_depth, max_gmm_fit_depth,
             min_kl_divergence, fdr_threshold,
             processes):
    res = parallel_test(
        cntrl_hdf5_fns, treat_hdf5_fns,
        min_depth, max_gmm_fit_depth,
        min_kl_divergence, fdr_threshold,
        processes
    )
    to_bed(res, output_bed_fn, fdr_threshold)


if __name__ == '__main__':
    gmmtest()