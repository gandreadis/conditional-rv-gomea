export PYTHONPATH="$PYTHONPATH:."

PROBLEM=$1
OPTIONS=("full" "univariate" "lt-static-gbo,lt-fb-online-unpruned,lt-fb-online-pruned" "ucond-gg-gbo,ucond-fg-gbo,ucond-hg-gbo,mcond-hg-gbo" "ucond-gg-fb,ucond-fg-fb,ucond-hg-fb,mcond-hg-fb" "ucond-gg-fb-generic,ucond-fg-fb-generic,ucond-hg-fb-generic,mcond-hg-fb-generic")

for option in $OPTIONS;
do
  echo "-========================-"
  echo $option
  echo "-========================-"
  python rvgomea/cmd/run_set_of_bisections.py -o data/${PROBLEM}_bbo -p $PROBLEM -b -l $option -d 20,40,80 -r 5
done

