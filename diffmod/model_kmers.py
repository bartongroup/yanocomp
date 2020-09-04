import logging
import random
import itertools as it
from functools import partial
from contextlib import contextmanager
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
import pomegranate as pm
import h5py as h5

import click


logger = logging.getLogger('gmmtest')


MODEL_COLUMNS = [
    'kmer',
    'current_skew', 'current_mean', 'current_std', 'current_model_ks_stat',
    'dwell_skew', 'dwell_mean', 'dwell_std', 'dwell_model_ks_stat',
]


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


def get_expressed_genes(hdf5_fns):
    '''
    Identify the genes which have eventaligned reads in all samples
    '''
    # filter out any transcripts that are not expressed in all samples
    with hdf5_list(hdf5_fns) as datasets:
        genes = set(datasets[0].keys())
        for d in datasets[1:]:
            genes.intersection_update(set(d.keys()))
    return list(genes)


def fetch_gene_kmers(gene_ids, hdf5_fns, max_events_per_pos):
    gene_events = []
    with hdf5_list(hdf5_fns) as datasets:
        for d in datasets:
            for g_id in gene_ids:
                event_table = pd.DataFrame(d[f'{g_id}/events'][:])[
                    ['pos', 'mean', 'duration']
                ]
                # should help to prevent highly expressed genes/positions
                # from dominating the distribution
                event_table = event_table.groupby('pos').sample(
                    n=max_events_per_pos, replace=True
                )
                # increase precision and log duration...
                event_table['mean'] = event_table['mean'].astype(np.float64)
                event_table['duration'] = np.log10(
                    event_table['duration'].astype(np.float64)
                )
                kmers = dict(d[f'{g_id}/kmers'][:])
                event_table['kmer'] = event_table.pos.map(kmers)
                gene_events.append(event_table[['kmer', 'mean', 'duration']])
    return pd.concat(gene_events)


def fetch_random_kmer_profiles(hdf5_fns, pool,
                               n_genes=1_000,
                               max_events_per_pos=5,
                               max_events_per_kmer=10_000):
    kmer_profiles = {
        ''.join(k).encode(): [] for k in it.product('ACGT', repeat=5)
    }
    gene_ids = random.sample(get_expressed_genes(hdf5_fns), n_genes)
    gene_id_chunks = np.array_split(gene_ids, pool._processes)
    all_events = []
    _fetch_func = partial(
        fetch_gene_kmers,
        hdf5_fns=hdf5_fns,
        max_events_per_pos=max_events_per_pos
    )
    all_events = pool.map(_fetch_func, gene_id_chunks)
    all_events = pd.concat(all_events)
    event_sample = all_events.groupby('kmer').sample(
        n=max_events_per_kmer, replace=True
    )
    return event_sample


def dbscan_filter(group, eps=2.5, min_sample_frac=0.1):
    kmer, events = group
    min_samples = int(len(events) * min_sample_frac)
    pred = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(
        events[['mean', 'duration']]
    ).ravel()
    return events[pred != -1]


def remove_outliers(kmer_events, pool, eps=2.5, min_sample_frac=0.1):
    _filter_func = partial(dbscan_filter, eps=eps, min_sample_frac=min_sample_frac)
    cleaned = pool.map(_filter_func, kmer_events.groupby('kmer'))
    return pd.concat(cleaned)


def ks_goodness_of_fit(data, model):
    model_sp = stats.norm(*model.parameters).cdf
    ks_stat, p_val = stats.ks_1samp(data, model_sp)
    return ks_stat, p_val


def fit_models(kmer_events):
    res = []
    for kmer, events in kmer_events.groupby('kmer'):
        current_skew = stats.skew(events['mean'])
        current_model = pm.NormalDistribution.from_samples(
            events['mean'].values.reshape(-1, 1)
        )
        current_ks_stat, _ = ks_goodness_of_fit(
            events['mean'], current_model
        )
        dwell_skew = stats.skew(events['duration'])
        dwell_model = pm.NormalDistribution.from_samples(
            events['duration'].values.reshape(-1, 1)
        )
        dwell_ks_stat, _ = ks_goodness_of_fit(
            events['duration'], dwell_model
        )
        res.append([kmer.decode(),
                    current_skew, *current_model.parameters, current_ks_stat,
                    dwell_skew, *dwell_model.parameters, dwell_ks_stat])
    res = pd.DataFrame(res, columns=MODEL_COLUMNS).set_index('kmer')
    return res


def calculcate_model_priors(hdf5_fns,
                            n_genes=1_000,
                            max_events_per_pos=5,
                            max_events_per_kmer=10_000,
                            dbscan_eps=2.5,
                            dbscan_min_frac=0.1,
                            processes=1):
    with mp.Pool(processes) as pool:
        kmer_events = fetch_random_kmer_profiles(
            hdf5_fns, pool,
            n_genes,
            max_events_per_pos,
            max_events_per_kmer
        )
        kmer_events = remove_outliers(kmer_events, pool)
    model = fit_models(kmer_events)
    return model


@click.command()
@click.option('-h', '--hdf5-fns', required=True, multiple=True)
@click.option('-o', '--model-output-fn', required=True)
@click.option('-g', '--n-genes-sampled', required=False, default=1_000)
@click.option('-p', '--max-events-sampled-per-pos', required=False, default=5)
@click.option('-k', '--events-sampled-per-kmer', required=False, default=10_000)
@click.option('--dbscan-eps', required=False, default=2.5)
@click.option('--dbscan-min-frac', required=False, default=0.1)
@click.option('-p', '--processes', required=False, default=1)
def model_priors(hdf5_fns, model_output_fn,
                 n_genes_sampled, max_events_sampled_per_pos,
                 events_sampled_per_kmer,
                 dbscan_eps, dbscan_min_frac,
                 processes):
    model = calculcate_model_priors(
        hdf5_fns,
        n_genes=n_genes_sampled,
        max_events_per_pos=max_events_sampled_per_pos,
        max_events_per_kmer=events_sampled_per_kmer,
        dbscan_eps=dbscan_eps,
        dbscan_min_frac=dbscan_min_frac,
        processes=processes
    )
    model.to_csv(
        model_output_fn,
        sep='\t',
        float_format='%.3f'
    )
