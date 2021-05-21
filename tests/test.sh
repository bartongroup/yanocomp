set -e;

FAILURES=0;

check()
{
    if diff $1 $2; then
        echo pass
    else
        FAILURES=$(expr $FAILURES + 1);
        echo fail
    fi
}


yanocomp prep -e test_data/hek293t_wt.tsv.gz -h cntrl.collapsed.h5
yanocomp prep -e test_data/hek293t_mettl3.tsv.gz -h treat.collapsed.h5

yanocomp gmmtest -c cntrl.collapsed.h5 -t treat.collapsed.h5 -o treat_v_cntrl.bed


check treat_v_cntrl.bed test_data/hek293t_wt_vs_mettl13.bed
rm cntrl.collapsed.h5 treat.collapsed.h5 treat_v_cntrl.bed

[[ $FAILURES -eq 0 ]] || exit 1;