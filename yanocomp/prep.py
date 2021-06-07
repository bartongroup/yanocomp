import os
import logging
import itertools as it
from operator import itemgetter
from bisect import bisect_right
from functools import partial
import multiprocessing as mp

import numpy as np
import click

from .opts import make_dataclass_decorator
from .io import parse_eventalign, load_gtf_database, save_events_to_hdf5

logger = logging.getLogger('yanocomp')


def transcript_to_genomic(pos, genomic_mapping):
    chrom = genomic_mapping['chrom']
    gene_id = genomic_mapping['gene_id']
    strand = genomic_mapping['strand']
    transcript_length = genomic_mapping['ln']
    invs = genomic_mapping['invs']
    inv_cs = genomic_mapping['inv_cs']
    if not 0 <= pos < transcript_length:
        raise ValueError('position out of range')
    if strand == '-':
        pos = transcript_length - pos - 1
    exon_idx = bisect_right(inv_cs, pos) - 1
    genomic_pos = invs[exon_idx][0] + (pos - inv_cs[exon_idx])
    return chrom, genomic_pos, strand, gene_id


def calculate_kmer_level_stats(means, stdvs, ns):
    '''
    calculates the kmer mean and std given a list of
    event means, stds, and the number of samples per event
    '''
    var = stdvs ** 2
    pooled_var = sum(var * ns) / sum(ns)
    pooled_var_correction = 0
    for i, j in it.combinations(np.arange(len(var)), r=2):
        pooled_var_correction += ns[i] * ns[j] * (means[i] - means[j]) ** 2
    pooled_var_correction /= sum(ns) ** 2
    pooled_std = np.sqrt(pooled_var + pooled_var_correction)
    pooled_mean = sum(means * ns) / sum(ns)
    return pooled_mean, pooled_std


def collapse_chunk(ea_chunk, to_gene=False):
    '''
    collapse a set of eventalign records to kmer level means, stds, and
    durations
    '''
    ea_chunk, gtf_chunk = ea_chunk
    eventalign_grouper = itemgetter('t_id', 'pos', 'kmer', 'r_id')
    ea_chunk_iter = it.groupby(ea_chunk, eventalign_grouper)

    collapsed = []
    gene_info = {}
    for (transcript_id, tpos, kmer, read_id), records in ea_chunk_iter:

        if to_gene:
            chrom, gpos, strand, gene_id = transcript_to_genomic(
                tpos, gtf_chunk[transcript_id]
            )
        else:
            gene_id = transcript_id
            gpos = tpos
            chrom = '.'
            strand = '.'
        gene_info[gene_id] = (chrom, strand)

        kmer_means = []
        kmer_stds = []
        kmer_lengths = []
        kmer_duration = 0
        for event in records:
            kmer_means.append(event['mean'])
            kmer_stds.append(event['std'])
            kmer_lengths.append(event['points'])
            kmer_duration += event['duration']

        # no need to do the stats if there is only one event
        if len(kmer_means) == 1:
            kmer_mean = kmer_means[0]
            kmer_std = kmer_stds[0]
        else:
            kmer_mean, kmer_std = calculate_kmer_level_stats(
                np.asarray(kmer_means),
                np.asarray(kmer_stds),
                np.asarray(kmer_lengths)
            )
        collapsed.append([gene_id, transcript_id, read_id, gpos,
                          kmer, kmer_mean, kmer_std, kmer_duration])
    # sort chunks by gene_id to speed up writing to disk
    collapsed.sort(key=itemgetter(0))
    return collapsed, gene_info


def get_gtf_chunk(chunk_transcript_ids, gtf):
    try:
        gtf_chunk = {t_id: gtf[t_id] for t_id in chunk_transcript_ids}
    except TypeError:
        # gtf is None
        gtf_chunk = None
    return gtf_chunk


def get_chunks(ea, approx_chunksize, gtf=None):
    '''
    chunks eventalign records. all records from the same read should be
    included in the same chunk (so long as they are contiguous in the file)
    '''
    ea_chunk = []
    curr_read_id = None
    chunk_transcript_ids = set()
    for record in ea:
        read_id = record['r_id']
        transcript_id = record['t_id']
        if read_id != curr_read_id and len(ea_chunk) >= approx_chunksize:
            gtf_chunk = get_gtf_chunk(chunk_transcript_ids, gtf)
            yield ea_chunk, gtf_chunk
            ea_chunk = []
            chunk_transcript_ids = set()
        curr_read_id = read_id
        ea_chunk.append(record)
        chunk_transcript_ids.add(transcript_id)
    else:
        gtf_chunk = get_gtf_chunk(chunk_transcript_ids, gtf)
        yield ea_chunk, gtf_chunk


def parallel_collapse(ea_fn, gtf_fn, processes, chunksize):
    '''
    collapses the records from the eventalign file
    '''
    if gtf_fn is not None:
        logger.info(f'Loading GTF records from {os.path.abspath(gtf_fn)}')
        gtf = load_gtf_database(
            gtf_fn, mapped_to='transcript_id', parent_id='gene_id'
        )
        to_gene = True
        logger.info(f'{len(gtf):,} transcript records loaded from GTF')
    else:
        logger.info('No GTF provided, records will be grouped by transcript')
        gtf = None
        to_gene = False

    ea_chunker = get_chunks(parse_eventalign(ea_fn), chunksize, gtf)
    with mp.Pool(processes) as pool:
        _collapse_func = partial(collapse_chunk, to_gene=to_gene)
        for chunk, gene_info in pool.imap_unordered(_collapse_func, ea_chunker):
            # group by gene
            for gene_id, records in it.groupby(chunk, itemgetter(0)):
                chrom, strand = gene_info[gene_id]
                yield gene_id, chrom, strand, records


@click.command(short_help='Convert nanopolish eventalign tsv to HDF5')
@click.option('-e', '--eventalign-fn', required=False, default='-', show_default=True,
              help=('File containing output of nanopolish eventalign. '
                    'Can be gzipped. Use - to read from stdin'))
@click.option('-h', '--hdf5-fn', required=True,
              help='Output HDF5 file')
@click.option('-g', '--gtf-fn', required=False, default=None,
              help=('Optional GTF file which can be used to convert '
                    'transcriptomic coordinates to genomic'))
@click.option('-p', '--processes', required=False, default=1, show_default=True)
@click.option('-n', '--chunksize', required=False, default=1_000_000, hidden=True)
@make_dataclass_decorator('CollapseOpts')
def nanopolish_collapse(opts):
    '''
    Parse nanopolish eventalign tabular data into a more managable
    HDF5 format...
    '''
    save_events_to_hdf5(
        parallel_collapse(
            opts.eventalign_fn,
            opts.gtf_fn,
            opts.processes,
            opts.chunksize
        ),
        opts.hdf5_fn
    )


if __name__ == '__main__':
    nanopolish_collapse()