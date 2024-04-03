export PYTHONPATH="$PYTHONPATH:."

PROBLEMS=("sphere" "rosenbrock" "reb2-chain-weak" "reb2-chain-strong" "reb2-chain-alternating" "reb5-no-overlap" "reb5-small-overlap" "reb5-small-overlap-alternating" "osoreb" "reb5-large-overlap" "reb5-disjoint-pairs" "reb-grid")
#PROBLEMS=("sphere" "rosenbrock" "reb2-chain-weak" "reb2-chain-strong" "reb2-chain-alternating" "reb5-no-overlap" "reb5-small-overlap" "reb5-small-overlap-alternating" "osoreb" "osoreb-big-strong" "osoreb-small-strong" "reb5-large-overlap" "reb5-disjoint-pairs" "reb-grid")
#FRAGMENTS=("snellius3") #"shark" "snellius1" "snellius2")
#FRAGMENTS=("snellius3" "shark2" "shark3")
FRAGMENTS=("shark3")

for problem in "${PROBLEMS[@]}"
do
    mkdir gecco-data/scalability-bisection-${problem}

    for fragment in "${FRAGMENTS[@]}"
    do
        rsync -P -a gecco-data/scalability-all-results/${fragment}/data/scalability-bisection-${problem}/ gecco-data/scalability-bisection-${problem}/
    done
done

for problem in "${PROBLEMS[@]}"
do
  echo ${problem}
  python rvgomea/cmd/aggregate_set_of_bisections.py gecco-data/scalability-bisection-${problem}
  echo ""
done
