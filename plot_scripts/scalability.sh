export PYTHONPATH="$PYTHONPATH:."

PROBLEMS=("sphere" "rosenbrock" "summation-cancellation" "reb2-chain-weak" "reb2-chain-strong" "reb2-chain-alternating" "reb5-no-overlap" "reb5-small-overlap" "reb5-small-overlap-alternating" "reb5-large-overlap" "reb5-disjoint-pairs" "reb-grid")

#for problem in "${PROBLEMS[@]}"
#do
#  echo ${problem}
#  python rvgomea/cmd/aggregate_set_of_bisections.py gecco-data/scalability-bisection-${problem}
#  echo ""
#done

python rvgomea/plots/gecco/plot_bisection_scalability.py gecco-data/scalability-bisection- sphere,rosenbrock,summation-cancellation,reb2-chain-weak,reb2-chain-strong,reb2-chain-alternating,reb5-no-overlap,reb5-small-overlap,reb5-small-overlap-alternating,reb5-large-overlap,reb5-disjoint-pairs,reb-grid "(a) Sphere,(b) Rosenbrock,(c) SummationCancellation,(d) REB2ChainWeak,(e) REB2ChainStrong,(f) REB2ChainAlternating,(g) REB5NoOverlap,(h) REB5SmallOverlap,(i) REB5SmallOverlapAlternating,(j) REB5LargeOverlap,(k) REB5DisjointPairs,(l) REBGrid"
