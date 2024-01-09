PROBLEMS=("sphere" "rosenbrock" "reb2-chain-weak" "reb2-chain-strong" "reb2-chain-alternating" "reb5-no-overlap" "reb5-small-overlap" "reb5-small-overlap-alternating" "osoreb" "reb5-large-overlap" "reb5-disjoint-pairs" "reb-grid")
FRAGMENTS=("meteor" "shark" "snellius1" "snellius2")

for problem in "${PROBLEMS[@]}"
do
    mkdir gecco-data/scalability-bisection-${problem}

    for fragment in "${FRAGMENTS[@]}"
    do
        rsync -P -a gecco-data/scalability-all-results/${fragment}/scalability-bisection-${problem}/ gecco-data/scalability-bisection-${problem}/
    done
done
