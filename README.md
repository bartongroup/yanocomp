# yanocomp
### (yet another nanopore modification comparison tool)

`yanocomp` does detection of RNA modifications from nanopore direct RNA sequencing (DRS) data which has been "eventaligned" using [`nanopolish`](https://github.com/jts/nanopolish). It uses a comparative approach with General mixture models similar to those used by [`nanocompore`](https://github.com/tleonardi/nanocompore) and [`xpore`](https://github.com/GoekeLab/xpore). The main selling points of `yanocomp` are:

* It parses nanopolish eventalign output on the fly into a (relatively) compact HDF5 file allowing random access.
* A GTF file can be provided to convert transcriptomic coordinates from eventalign back to genomic coordinates.
* It uses [`pomegranate`](https://github.com/jmschrei/pomegranate) for model fitting (which makes it fast!).
* It fits models using multiple adjacent kmers to better separate modified and unmodified distributions.
* A uniform distribution is used to model outliers caused by low quality signal or poor alignment. This improves the fit and reduces false positives.

For the comparative method, you will need nanopore DRS data from a control sample with normal levels of modifications and a treatment sample with altered levels of modifications. NB: `yanocomp` is still in quite a beta stage so there are likely to be bugs!

## Installation:

`yanocomp` has been tested with python 3.8.5, and requires `numpy`, `pandas`, `h5py`, `pomegranate`, `scipy`, `statsmodels`, and `click`. The easiest way to install it is using the conda environment yaml provided:

```
git clone https://github.com/bartongroup/yanocomp.git
cd yanocomp
conda env create -f yanocomp.yml
conda activate yanocomp
```

Alternatively `yanosim` and the required packages can be installed using pip:

```
pip install git+git://github.com/bartongroup/yanocomp.git
```

## Usage:

Before running `yanocomp`, you should align your nanopore DRS data to a transcriptome reference (with no spliced alignment, no secondary alignments), for example using [`minimap2`](https://github.com/lh3/minimap2). You should then align the signal level data using [`nanopolish`](https://github.com/jts/nanopolish) with the command:

```
nanopolish eventalign \
  --scale-events \
  --signal-index \
  --print-read-names \
  -r <fastq-fn> \
  -b <bam-fn> \
  -g <reference-fasta-fn>
```

`yanocomp` itself has two main commands:

### `prep`:

`yanocomp prep` will parse a text or gzipped file containing the tabular results from `nanopolish eventalign` and summarise them into a more compact HDF5 file. Output from eventalign can also be piped to `yanocomp` prep on the fly. Basecalled reads should be aligned to a  before event alignment, but `yanocomp prep` can convert back to genomic coordinates if a GTF file is provided.

#### Options:
```
$ yanocomp prep --help
Usage: yanocomp prep [OPTIONS]

  Parse nanopolish eventalign tabular data into a more managable HDF5
  format...

Options:
  -e, --eventalign-fn TEXT  File containing output of nanopolish eventalign.
                            Can be gzipped. Use - to read from stdin
                            [default: -]

  -h, --hdf5-fn TEXT        Output HDF5 file  [required]
  -g, --gtf-fn TEXT         Optional GTF file which can be used to convert
                            transcriptomic coordinates to genomic

  -p, --processes INTEGER   [default: 1]
  --help                    Show this message and exit.
```

#### Output:

A HDF5 file storing summarised event level signal data. The file contains one group per gene (if a GTF is provided) or transcript (if no GTF is provided).

## `gmmtest`:

`yanocomp gmmtest` compares samples from two conditions to identify positions with differences in the profile of aligned nanopore signal "events". Positions where there are detectable differences in signal (detected with Kolmogorov Smirnov test, default p<0.05 and KS statistic > 0.1) will progress to model fitting. This is done by fitting a two-gaussian general mixture model (plus a third uniform distribution for outliers) using `pomegranate` and then estimating the fraction of modified and unmodified reads in each sample. These estimates are used to perform a G-test. Fitted models can be optionally used to make single-molecule modification predictions.

#### Options:
```
$ yanocomp gmmtest --help
Usage: yanocomp gmmtest [OPTIONS]

  Differential RNA modifications using nanopore DRS signal level data

Options:
  -c, --cntrl-hdf5-fns TEXT       Control HDF5 files. Can specify multiple
                                  files using multiple -c flags  [required]

  -t, --treat-hdf5-fns TEXT       Treatment HDF5 files. Can specify multiple
                                  files using multiple -t flags  [required]

  -o, --output-bed-fn TEXT        Output bed file name  [required]
  -s, --output-sm-preds-fn TEXT   JSON file to output single molecule
                                  predictions. Can be gzipped (detected from
                                  name)

  -m, --prior-model-fn TEXT       Model file with expected kmer current
                                  distributions

  --test-level [gene|transcript]  Test at transcript level or aggregate to
                                  gene level  [default: gene]

  -w, --window-size INTEGER       How many adjacent kmers to model over
                                  [default: 5]

  -e, --outlier-factor FLOAT      Scaling factor for labelling outliers during
                                  model initialisation. Smaller means more
                                  aggressive labelling of outliers  [default:
                                  0.5]

  -n, --min-read-depth TEXT       Minimum reads per replicate to test a
                                  position. Default is to set dynamically

  -d, --max-fit-depth INTEGER     Maximum number of reads per replicate used
                                  to fit the model  [default: 1000]

  -k, --min-ks FLOAT              Minimum KS test statistic to attempt to
                                  build a model for a position  [default: 0.1]

  -f, --fdr-threshold FLOAT       False discovery rate threshold for output
                                  [default: 0.05]

  -p, --processes INTEGER RANGE   [default: 1]
  --help                          Show this message and exit.
```

#### Output:

A 19-column BED file format with the following values:
```
1. chrom [string]
2. start [integer]
3. end [integer]
4. gene_id:kmer [string]
5. score (min(- log10(FDR), 100)) [integer]
6. strand [string, either '+' or '-']
7. modification log ratio [float]
8. G-test p-value [float]
9. G-test FDR [float]
10. Control sample estimated fraction modified [float]
11. Treatment sample estimated fraction modified [float]
12. G-statistic (Treat vs cntrl) [float]
13. Homogeneity G-statistic [float]
14. Unmodified distribution mean [float]
15. Unmodified distribution standard deviation [float]
16. Modified distribution mean [float]
17. Modified distribution standard deviation [float]
18. Modified distribution shift direction [string, 'l' for lower or 'h' for higher]
19. KS statistic [float]
```

`gmmtest` can also optionally output a json file containing the single molecule modification predictions. The schema may change but is currently:

```
{
    "input_fns": {
        "cntrl": {
            "$rep": "<file_name>"
        },
        "treat": {
            "$rep": "<file_name>"
        }
    }
    "single_molecule_predictions": {
        "$gene_id": {
            "$pos": {
                "kmers": [...], # array of strings, shape (window_size,)
                "cntrl": {
                    "$rep": {
                        "read_ids": [...], # array of strings, shape (nreads,)
                        "events": [...], # array of floats, shape (nreads, window_size)
                        "preds": [...], # array of floats, shape (nreads,)
                    } 
                },
                "treat": {
                    "$rep": {
                        "read_ids": [...],
                        "events": [...],
                        "preds": [...],
                    }
                }
            }
        }
    }
}
```

