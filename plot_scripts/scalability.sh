export PYTHONPATH="$PYTHONPATH:."

PROBLEMS=("sphere" "rosenbrock" "reb2-chain-weak" "reb2-chain-strong" "reb2-chain-alternating" "reb5-no-overlap" "reb5-small-overlap" "reb5-small-overlap-alternating" "osoreb" "reb5-large-overlap" "reb5-disjoint-pairs" "reb-grid")

for problem in "${PROBLEMS[@]}"
do
  echo ${problem}
  python rvgomea/cmd/aggregate_set_of_bisections.py gecco-data/scalability-bisection-${problem}
  echo ""
done

python rvgomea/plots/gecco/plot_bisection_scalability.py gecco-data/scalability-aggregated/scalability-bisection- sphere,rosenbrock,reb2-chain-weak,reb2-chain-strong,reb2-chain-alternating,reb5-no-overlap,reb5-small-overlap,osoreb,reb5-large-overlap,reb5-small-overlap-alternating,reb5-disjoint-pairs,reb-grid "(a) Sphere,(b) Rosenbrock,(c) REB2Weak,(d) REB2Strong,(e) REB2Alternating,(f) REB5NoOverlap,(g) REB5SmallOverlap,(h) OSoREB,(i) REB5LargeOverlap,(j) REB5Alternating,(k) REB5DisjointPairs,(l) REBGrid" univariate,uni-hg-gbo-without_clique_seeding-conditional,mp-hg-gbo-without_clique_seeding-conditional,mp-hg-gbo-with_clique_seeding-conditional,lt-fb-online-pruned,uni-hg-fb_no_order-without_clique_seeding-conditional,mp-hg-fb_no_order-without_clique_seeding-conditional,mp-hg-fb_no_order-with_clique_seeding-conditional,vkd-cma

#python rvgomea/plots/gecco/plot_bisection_scalability.py gecco-data/scalability-aggregated/scalability-bisection- osoreb,osoreb-big-strong,osoreb-small-strong "(a) OSoREB,(b) OSoREBBigBlocksStrong,(c) OSoREBSmallBlocksStrong" univariate,uni-hg-gbo-without_clique_seeding-conditional,mp-hg-gbo-without_clique_seeding-conditional,mp-hg-gbo-with_clique_seeding-conditional,lt-fb-online-pruned,uni-hg-fb_no_order-without_clique_seeding-conditional,mp-hg-fb_no_order-without_clique_seeding-conditional,mp-hg-fb_no_order-with_clique_seeding-conditional,vkd-cma
