import dataclasses
import pysam
import click


def find_max_indel_size(aln, include_refskip=True):
    indel_ops = (1, 2, 3) if include_refskip else (1, 2)
    indel_sizes = []
    for op, ln in aln.cigar:
        if op in indel_ops:
            indel_sizes.append(ln)
    if len(indel_sizes):
        return max(indel_sizes)
    else:
        return 0


def filter_bam(bam, opts):
    for aln in bam.fetch():
        mapped_length = aln.query_alignment_length
        if mapped_length < opts.min_mapped_length:
            continue

        basecalled_length = aln.query_length
        if basecalled_length == 0:
            basecalled_length = aln.infer_query_length()
        assert basecalled_length >= mapped_length
        mapped_over_basecalled = mapped_length / basecalled_length
        if mapped_over_basecalled < opts.min_mapped_length_over_read_length:
            continue

        gap_comp_div = aln.get_tag('de')
        if gap_comp_div > opts.max_gap_compressed_divergence:
            continue

        mapq = aln.mapping_quality
        if mapq < opts.min_mapq:
            continue

        max_indel_size = find_max_indel_size(aln)
        if max_indel_size > opts.max_indel_size:
            continue

        yield aln


def write_output(output_stream, output_bam_fn, template_bam):
    with pysam.AlignmentFile(output_bam_fn, 'wb',
                             template=template_bam) as output_bam:
        for aln in output_stream:
            output_bam.write(aln)


@dataclasses.dataclass
class FilterOpts:
    bam_fn: str
    output_bam_fn: str
    min_mapped_length: int
    min_mapped_length_over_read_length: float
    max_indel_size: float
    max_gap_compressed_divergence: float
    min_mapq: int


def make_dataclass_decorator(dc):
    def _dataclass_decorator(cmd):
        @click.pass_context
        def _make_dataclass(ctx, **kwargs):
            return ctx.invoke(cmd, dc(**kwargs))
        return _make_dataclass
    return _dataclass_decorator


@click.command()
@click.option('-b', '--bam-fn', required=True)
@click.option('-o', '--output-bam-fn', required=True)
@click.option('-L', '--min-mapped-length', default=100)
@click.option('-r', '--min-mapped-length-over-read-length', default=0.85)
@click.option('-i', '--max-gap-compressed-divergence', default=0.15)
@click.option('-q', '--min-mapq', default=1)
@click.option('-d', '--max-indel-size', default=15)
@make_dataclass_decorator(FilterOpts)
def cli(opts):
    with pysam.AlignmentFile(opts.bam_fn) as bam:
        output_stream = filter_bam(bam, opts)
        write_output(output_stream, opts.output_bam_fn, bam)
    pysam.index(opts.output_bam_fn)


if __name__ == '__main__':
    cli()