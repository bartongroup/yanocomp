import os
import sys
import re
import gzip
import csv
import logging
from contextlib import contextmanager
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import h5py as h5

logger = logging.getLogger('diffmod')


# Functions for parsing input text files e.g. eventalign tsv or GTF

@contextmanager
def path_or_stdin_reader(fn):
    '''
    Read from stdin, or a gzipped or ungzipped text file
    '''
    if fn == '-':
        handle = sys.stdin
        decode_method = str
    elif os.path.splitext(fn)[1] == '.gz':
        handle = gzip.open(fn)
        decode_method = bytes.decode
    else:
        handle = open(fn)
        decode_method = str

    try:
        gen = (decode_method(line) for line in handle)
        yield gen
    finally:
        handle.close()
        gen.close()


def parse_eventalign(eventalign_fn):
    with path_or_stdin_reader(eventalign_fn) as handle:
        ea_parser = csv.DictReader(handle, delimiter='\t')
        fieldnames = set(ea_parser.fieldnames)
        if 'read_name' not in fieldnames:
            raise KeyError(
                'nanopolish must be run with --print-read-names option'
            )
        if not fieldnames.issuperset(['start_idx', 'end_idx']):
            raise KeyError(
                'cannot find "start_idx" or "end_idx" fields, '
                'nanopolish must be run with --signal-index option'
            )
        for record in ea_parser:
            parsed = {}
            parsed['t_id'] = record['contig']
            parsed['pos'] = int(record['position']) + 2
            parsed['kmer'] = record['reference_kmer']
            parsed['r_id'] = record['read_name']
            parsed['mean'] = float(record['event_level_mean'])
            parsed['std'] = float(record['event_stdv'])
            parsed['duration'] = float(record['event_length'])
            parsed['points'] = (int(record['end_idx']) -
                                int(record['start_idx']))
            yield parsed


def ignore_comments(handle, comment_char='#'):
    for line in handle:
        if not line.startswith(comment_char):
            yield line


def parse_gtf_attrs(attrs):
    attrs = re.findall(r'(\w+) \"(.+?)\"(?:;|$)', attrs)
    return dict(attrs)


def read_gtf(gtf_fn, use_ftype):
    with path_or_stdin_reader(gtf_fn) as handle:
        gtf_parser = csv.DictReader(
            ignore_comments(handle),
            delimiter='\t',
            fieldnames=['chrom', 'source', 'ftype',
                        'start', 'end', 'score',
                        'strand', 'frame', 'attrs']
        )
        for record in gtf_parser:
            ftype = record['ftype']
            if ftype == use_ftype:
                chrom = record['chrom']
                start, end = int(record['start']) - 1, int(record['end'])
                strand = record['strand']
                attrs = parse_gtf_attrs(record['attrs'])
                yield chrom, start, end, ftype, strand, attrs


def load_gtf_database(gtf_fn, mapped_to='transcript_id', parent_id='gene_id'):
    gtf = {}
    for chrom, start, end, ftype, strand, attrs in read_gtf(gtf_fn, 'exon'):
        transcript_id = attrs[mapped_to]
        gene_id = attrs[parent_id]
        if transcript_id not in gtf:
            gtf[transcript_id] = {
                'gene_id': gene_id,
                'chrom': chrom,
                'strand': strand,
                'invs': []
            }
        gtf[transcript_id]['invs'].append((start, end))
    for transcript in gtf.values():
        transcript['invs'].sort()
        transcript['start'] = transcript['invs'][0][0]
        transcript['invs_lns'] = [e - s for s, e in transcript['invs']]
        transcript['ln'] = sum(transcript['invs_lns'])
        transcript['inv_cs'] = []
        i = 0
        for ln in transcript['invs_lns']:
            transcript['inv_cs'].append(i)
            i += ln
    return gtf


# functions for creating/loading data from diffmod's HDF5 layout.

def incrementing_index(index, identifier):
    try:
        idx = index[identifier]
    except KeyError:
        idx = len(index)
        index[identifier] = idx
    return idx


EVENT_DTYPE = np.dtype([
    ('transcript_idx', np.uint8),
    ('read_idx', np.uint32),
    ('pos', np.uint32),
    ('mean', np.float16),
    ('std', np.float16),
    ('duration', np.float16)
])

KMER_DTYPE = np.dtype([
    ('pos', np.uint32),
    ('kmer', h5.string_dtype(length=5))
])

TRANSCRIPT_DTYPE = h5.string_dtype()

READ_DTYPE = h5.string_dtype(length=32)


def save_events_to_hdf5(collapsed, hdf5_fn):
    kmer_index = defaultdict(dict)
    transcript_index = defaultdict(OrderedDict)
    read_index = defaultdict(OrderedDict)
    gene_attrs = {}
    with h5.File(hdf5_fn, 'w') as o:
        for gene_id, chrom, strand, records in collapsed:
            data = []
            gene_attrs[gene_id] = (chrom, strand)
            for _, t_id, r_id, pos, kmer, mean, std, duration in records:
                t_x = incrementing_index(transcript_index[gene_id], t_id)
                r_x = incrementing_index(read_index[gene_id], r_id)
                data.append((t_x, r_x, pos, mean, std, duration))
                kmer_index[gene_id][pos] = kmer
            data = np.asarray(data, dtype=EVENT_DTYPE)
            n_records = len(data)
            try:
                # if records for transcript are split across
                # multiple chunks we will need to extend the existing
                # dataset.
                output_dataset = o[f'{gene_id}/events']
                i = len(output_dataset)
                output_dataset.resize(i + n_records, axis=0)
            except KeyError:
                output_dataset = o.create_dataset(
                    f'{gene_id}/events',
                    shape=(n_records,),
                    maxshape=(None,),
                    dtype=EVENT_DTYPE,
                    chunks=True,
                    compression='gzip',
                )
                i = 0
            output_dataset[i: i + n_records] = data
        # now add kmers, read ids and transcript_ids
        for gene_id, kmer_pos in kmer_index.items():
            kmers = np.array(list(kmer_pos.items()), dtype=KMER_DTYPE)
            o.create_dataset(
                f'{gene_id}/kmers',
                data=kmers,
                dtype=KMER_DTYPE,
                compression='gzip'
            )
            transcript_ids = np.array(
                list(transcript_index[gene_id].keys()),
                dtype=TRANSCRIPT_DTYPE
            )
            o.create_dataset(
                f'{gene_id}/transcript_ids',
                data=transcript_ids,
                dtype=TRANSCRIPT_DTYPE,
                compression='gzip'
            )
            read_ids = np.array(
                list(read_index[gene_id].keys()),
                dtype=READ_DTYPE
            )
            o.create_dataset(
                f'{gene_id}/read_ids',
                data=read_ids,
                dtype=READ_DTYPE,
                compression='gzip'
            )
            chrom, strand = gene_attrs[gene_id]
            o[gene_id].attrs.create(
                'chrom', data=chrom, dtype=h5.string_dtype()
            )
            o[gene_id].attrs.create(
                'strand', data=strand, dtype=h5.string_dtype()
            )


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


def get_shared_keys(hdf5_handles):
    '''
    Identify the intersection of the keys in a list of hdf5 files
    '''
    # filter out any transcripts that are not expressed in all samples
    genes = set(hdf5_handles[0].keys())
    for d in hdf5_handles[1:]:
        genes.intersection_update(set(d.keys()))
    return list(genes)


def load_gene_kmers(gene_id, datasets):
    '''
    Get all recorded kmers from the different datasets
    '''
    kmers = {}
    # positions (and their kmers) which are recorded may vary across datasets
    for d in datasets:
        k = d[f'{gene_id}/kmers'][:].astype(
            np.dtype([('pos', np.uint32), ('kmer', 'U5')])
        )
        kmers.update(dict(k))
    return kmers


def load_gene_events(gene_id, datasets,
                     load_kmers=False,
                     load_transcript_ids=False,
                     load_read_ids=False):
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
        if load_transcript_ids:
            t = d[f'{gene_id}/transcript_ids'][:]
            t = {i: t_id for i, t_id in enumerate(t)}
            e['transcript_idx'] = e.transcript_idx.map(t)
        if load_read_ids:
            r = d[f'{gene_id}/read_ids'][:].astype('U32')
            r = {i: r_id for i, r_id in enumerate(r)}
            e['read_idx'] = e.read_idx.map(r)
        e['replicate'] = i
        gene_events.append(e)
    gene_events = pd.concat(gene_events)
    if load_kmers:
        kmers = load_gene_kmers(gene_id, datasets)
        gene_events['kmer'] = gene_events.pos.map(kmers)
    return gene_events


def load_gene_attrs(gene_id, datasets):
    '''
    Extracts important info i.e. chromosome, strand
    for a gene from the HDF5 files...
    '''
    # get general info which should be same for all datasets
    g = datasets[0][gene_id]
    chrom = g.attrs['chrom']
    strand = g.attrs['strand']
    return chrom, strand


# functions for loading/saving output from `priors` and `gmmtest` commands

def save_model_priors(model, model_output_fn):
    '''
    Save model to tsv
    '''
    model.to_csv(
        model_output_fn,
        sep='\t',
        float_format='%.3f'
    )


DEFAULT_PRIORS_MODEL_FN = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    'data/vir1_r9.4_model.tsv'
)


def load_model_priors(model_fn=None):
    '''
    Load the parameters for the expected kmer distributions.
    '''
    if model_fn is None:
        model_fn = DEFAULT_PRIORS_MODEL_FN
    m = pd.read_csv(
        model_fn, sep='\t', comment='#', index_col='kmer'
    )
    m = m[['current_mean', 'current_std', 'dwell_mean', 'dwell_std']]
    return m.T.to_dict(orient='list')


def save_gmmtest_results(res, output_bed_fn, fdr_threshold=0.05,
                         custom_filter=None):
    '''
    write main results to bed file
    '''
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
    sig_res = sig_res.sort_values(by=['chrom', 'pos'])
    logger.info(f'Writing output to {os.path.abspath(output_bed_fn)}')
    with open(output_bed_fn, 'w') as bed:
        for record in sig_res.itertuples(index=False):
            (chrom, pos, gene_id, kmer, strand,
             log_odds, pval, fdr, c_fm, t_fm,
             g_stat, hom_g_stat, kld) = record
            score = int(round(min(- np.log10(fdr), 100)))
            bed_record = (
                f'{chrom:s}\t{pos - 2:d}\t{pos + 3:d}\t'
                f'{gene_id}:{kmer}\t{score:d}\t{strand:s}\t'
                f'{log_odds:.2f}\t{pval:.2g}\t{fdr:.2g}\t'
                f'{c_fm:.2f}\t{t_fm:.2f}\t'
                f'{g_stat:.2f}\t{hom_g_stat:.2f}\t{kld:.2f}\n'
            )
            bed.write(bed_record)