export PYTHONPATH="$PYTHONPATH:."

# Desc:   max-overlapping-cliques vs trad. MP
# Models: mp-fb-online-fg,mp-fg-fb_no_order-with_clique_seeding-non_conditional,full
# ID:     overlapping-cliques

# Desc:   FG vs HG
# Models: mp-fb-online-fg,mp-fb-online-hg
# ID:     gom-non-overlapping
# (copy over / reuse one of first)

# Desc:   FG vs HG
# Models: mp-fg-fb_no_order-with_clique_seeding-non_conditional,mp-hg-fb_no_order-with_clique_seeding-non_conditional
# ID:     gom-overlapping
# (copy over / reuse one of first)

# Desc:   Conditional
# Models: uni-hg-gbo-without_clique_seeding-conditional,mp-hg-gbo-with_clique_seeding-conditional,uni-hg-fb_no_order-without_clique_seeding-conditional,mp-hg-fb_no_order-with_clique_seeding-conditional
# ID:     conditional
# (copy over / reuse one of first)

models=$1
id=$2
CONDITIONS=("1" "2" "3" "4" "5" "6")
ROTATIONS=("0" "5" "10" "15" "20" "25" "30" "35" "40" "45")

for c in "${CONDITIONS[@]}"
do
  echo "///// Condition number  ${c}"
  for r in "${ROTATIONS[@]}"
  do
    echo "///// Rotation angle  ${r}"
    python rvgomea/cmd/run_set_of_bisections.py -o data/reb-chain-bisection-${id} -p reb-chain-condition-${c}-rotation-${r} -l ${models} -d 20 -r 5
  done
done

echo "Done."
