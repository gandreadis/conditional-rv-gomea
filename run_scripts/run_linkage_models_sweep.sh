export PYTHONPATH="$PYTHONPATH:."

PROBLEM=$1
ID=$2
#LM="lt-static-gbo,lt-fb-online-unpruned,lt-fb-online-pruned"
LM="mcond-hg-gbo,mcond-hg-fb,mcond-hg-fb-generic"
#LM="mcond-hg-fb-generic,mcond-hg-fb"
#OPTIONS=("ucond-gg-gbo,ucond-fg-gbo,ucond-hg-gbo,mcond-hg-gbo" "ucond-gg-fb,ucond-fg-fb,ucond-hg-fb,mcond-hg-fb" "ucond-gg-fb-generic,ucond-fg-fb-generic,ucond-hg-fb-generic,mcond-hg-fb-generic")
OPTIONS=( "lt-static-gbo,lt-fb-online-unpruned,lt-fb-online-pruned" "ucond-gg-gbo,ucond-fg-gbo,ucond-hg-gbo,mcond-hg-gbo" "ucond-hg-fb,mcond-hg-fb" "ucond-hg-fb-generic,mcond-hg-fb-generic" "full" )

python rvgomea/cmd/run_sweep.py -o data/${PROBLEM}_${ID}_sweep_bbo -p $PROBLEM -b -l $LM -d 20 -r 30 -s 64
