import os
import sys
import re
import csv
import gzip
import itertools as it
from operator import itemgetter
from bisect import bisect_right
from collections import defaultdict, OrderedDict
from functools import partial
import multiprocessing as mp

import numpy as np
import h5py as h5
import click


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


class FileLike:

    def __init__(self, fn):
        self._fn = fn
        self._open()

    def close(self):
        self._handle.close()

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()


class UnzippedOrGzippedFile(FileLike):

    def _open(self):
        fn = self._fn
        if fn == '-':
            self._handle = sys.stdin
            self._decode_method = str
        elif os.path.splitext(fn)[1] == '.gz':
            self._handle = gzip.open(fn)
            self._decode_method = bytes.decode
        else:
            self._handle = open(fn)
            self._decode_method = str

    def __next__(self):
        return self._decode_method(next(self._handle))


class EventalignDictReader(FileLike):
    
    def _open(self):
        self._handle = UnzippedOrGzippedFile(self._fn)
        self._parser = csv.DictReader(self._handle, delimiter='\t')
        self._has_readname = 'read_name' in self._parser.fieldnames

    def __next__(self):
        r = next(self._parser)
        record = {}
        record['t_id'] = r['contig']
        record['pos'] = int(r['position']) + 2
        record['kmer'] = r['reference_kmer']
        try:
            record['r_id'] = r['read_name']
        except KeyError:
            raise KeyError('nanopolish must be run with --print-read-names option')
        record['mean'] = float(r['event_level_mean'])
        record['std'] = float(r['event_stdv'])
        record['duration'] = float(r['event_length'])
        try:
            record['points'] = int(r['end_idx']) - int(r['start_idx'])
        except KeyError:
            raise KeyError('cannot find "start_idx" or "end_idx" fields, '
                           'nanopolish must be run with --signal-index option')
        return record


def parse_gtf_attrs(attrs):
    attrs = re.findall('(\w+) \"(.+?)\"(?:;|$)', attrs)
    return dict(attrs)


def parse_gtf_record(record):
    chrom, _, ftype, start, end, _, strand, _, attrs = record.split('\t')
    attrs = parse_gtf_attrs(attrs)
    start, end = int(start) - 1, int(end)
    return chrom, start, end, ftype, strand, attrs


def load_gtf(gtf_fn, mapped_to='transcript_id', parent_id='gene_id'):
    gtf = {}
    with open(gtf_fn) as f:
        for record in f:
            if record.startswith('#'):
                continue
            chrom, start, end, ftype, strand, attrs = parse_gtf_record(record)
            if ftype == 'exon':
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


def transcript_to_genomic(pos, genomic_mapping):
    chrom = genomic_mapping['chrom']
    offset = genomic_mapping['start']
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
    collapse a set of eventalign records to kmer level means, stds, and durations
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
            offset = 0
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
            gtf_chunk = {t_id: gtf[t_id] for t_id in chunk_transcript_ids}
            yield ea_chunk, gtf_chunk
            ea_chunk = []
            chunk_transcript_ids = set()
        curr_read_id = read_id
        ea_chunk.append(record)
        chunk_transcript_ids.add(transcript_id)
    else:
        gtf_chunk = {t_id: gtf[t_id] for t_id in chunk_transcript_ids}
        yield ea_chunk, gtf_chunk


def parallel_collapse(ea_fn, gtf_fn, processes, chunksize):
    '''
    collapses the records from the eventalign file
    '''
    if gtf_fn is not None:
        gtf = load_gtf(gtf_fn, mapped_to='transcript_id', parent_id='gene_id')
        to_gene = True
    else:
        gtf = None
        to_gene = False
    with EventalignDictReader(ea_fn) as ea, mp.Pool(processes) as pool:
        collapse = partial(collapse_chunk, to_gene=to_gene)
        for chunk, gene_info in pool.imap_unordered(collapse, get_chunks(ea, chunksize, gtf)):
            # group by gene
            for gene_id, records in it.groupby(chunk, itemgetter(0)):
                chrom, strand = gene_info[gene_id]
                yield gene_id, chrom, strand, records


def incrementing_index(index, identifier):
    try:
        idx = index[identifier]
    except KeyError:
        idx = len(index)
        index[identifier] = idx
    return idx


def to_hdf5(collapsed, hdf5_fn):
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
            o[gene_id].attrs.create('chrom', data=chrom, dtype=h5.string_dtype())
            o[gene_id].attrs.create('strand', data=strand, dtype=h5.string_dtype())


@click.command()
@click.option('-e', '--eventalign-fn', required=False, default='-')
@click.option('-h', '--hdf5-fn', required=True)
@click.option('-g', '--gtf-fn', required=False, default=None)
@click.option('-p', '--processes', required=False, default=1)
@click.option('-n', '--chunksize', required=False, default=1_000_000)
def nanopolish_collapse(eventalign_fn, hdf5_fn, gtf_fn, processes, chunksize):
    to_hdf5(
        parallel_collapse(eventalign_fn, gtf_fn, processes, chunksize),
        hdf5_fn
    )


if __name__ == '__main__':
    collapse()