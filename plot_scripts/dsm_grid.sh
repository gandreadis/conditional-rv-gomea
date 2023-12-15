export PYTHONPATH="$PYTHONPATH:."

PROBLEMS=("sphere" "rosenbrock" "summation-cancellation" "reb2-chain-weak" "reb2-chain-strong" "reb2-chain-alternating" "reb5-no-overlap" "reb5-small-overlap" "reb5-small-overlap-alternating" "reb5-large-overlap" "reb5-disjoint-pairs" "reb-grid")
SIZES=(30 30 30 30 30 30 30 33 33 30 36 36)

for i in "${!PROBLEMS[@]}"
do
  problem=${PROBLEMS[i]}
  size=${SIZES[i]}
  echo ${problem}
  for i in {1..30}
  do
    python rvgomea/cmd/run.py -l mp-hg-fb_no_order-without_clique_seeding-conditional -b -p $problem -d $size -s 100 -i gecco-data/dsm_grid/${problem}/${i} -r ${i} -o -z
    echo ""
  done
  echo "///////"
done

python rvgomea/plots/gecco/plot_dsm_grid.py gecco-data/dsm_grid sphere,rosenbrock,summation-cancellation,reb2-chain-weak,reb2-chain-strong,reb2-chain-alternating,reb5-no-overlap,reb5-small-overlap,reb5-large-overlap,reb5-small-overlap-alternating,reb5-disjoint-pairs,reb-grid "(a) Sphere,(b) Rosenbrock,(c) SummationCancellation,(d) REB2Weak,(e) REB2Strong,(f) REB2Alternating,(g) REB5NoOverlap,(h) REB5SmallOverlap,(i) REB5LargeOverlap,(j) REB5Alternating,(k) REB5DisjointPairs,(l) REBGrid"
