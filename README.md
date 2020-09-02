# `diffmod`

`diffmod` does detection of RNA modifications from nanopore DRS data which has been "eventaligned" using `nanopolish`. It uses a comparative approach similar to that used by `nanocompore` and `xpore`. For the comparative method, you will need nanopore DRS data from a control sample with normal levels of modifications and a treatment sample with altered levels of modifications. There are two commands:

## `prep`:

`diffmod prep` will parse a text or gzipped file containing the tabular results from `nanopolish eventalign` and summarise them into a more compact HDF5 file. Output from eventalign can also be piped to diffmod prep on the fly.

## `test`:

`diffmod test` compares samples from two conditions to identify positions with differences in the profile of aligned nanopore signals. This is done by fitting a two-component gaussian mixture model using `pomegranate` and then estimating the fraction of modified and unmodified reads in each sample. These estimates are used to perform a G-test. Only positions where there is significant differences in signal (detected with t-test, default p<0.05) will have models fit, and only models with KL divergence over the threshold (default 0.5) will be tested for significant changes.