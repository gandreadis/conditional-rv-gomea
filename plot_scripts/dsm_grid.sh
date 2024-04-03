export PYTHONPATH="$PYTHONPATH:."

PROBLEMS=("sphere" "rosenbrock" "reb2-chain-weak" "reb2-chain-strong" "reb2-chain-alternating" "reb5-no-overlap" "reb5-small-overlap" "reb5-small-overlap-alternating" "osoreb" "reb5-large-overlap" "reb5-disjoint-pairs" "reb-grid")
SIZES=(30 30 30 30 30 30 33 33 30 30 36 36)
#PROBLEMS=("osoreb" "osoreb-big-strong" "osoreb-small-strong")
#SIZES=(30 30 30)

for i in "${!PROBLEMS[@]}"
do
  problem=${PROBLEMS[i]}
  size=${SIZES[i]}
  echo ${problem}
  for i in {1..30}
  do
    python rvgomea/cmd/run.py -l mp-hg-fb_no_order-with_clique_seeding-conditional -p $problem -d $size -s 250 -i gecco-data/dsm_grid/${problem}/${i} -r ${i} -o -z || exit 1
    echo ""
  done
  echo "///////"
done

python rvgomea/plots/gecco/plot_dsm_grid.py gecco-data/dsm_grid sphere,rosenbrock,reb2-chain-weak,reb2-chain-strong,reb2-chain-alternating,reb5-no-overlap,reb5-small-overlap,osoreb,reb5-large-overlap,reb5-small-overlap-alternating,reb5-disjoint-pairs,reb-grid "(a) Sphere,(b) Rosenbrock,(c) REB2Weak,(d) REB2Strong,(e) REB2Alternating,(f) REB5NoOverlap,(g) REB5SmallOverlap,(h) OSoREB,(i) REB5LargeOverlap,(j) REB5Alternating,(k) REB5DisjointPairs,(l) REBGrid"
#python rvgomea/plots/gecco/plot_dsm_grid.py gecco-data/dsm_grid osoreb,osoreb-big-strong,osoreb-small-strong "(a) OSoREB,(b) OSoREBBigBlocksStrong,(c) OSoREBSmallBlocksStrong"
