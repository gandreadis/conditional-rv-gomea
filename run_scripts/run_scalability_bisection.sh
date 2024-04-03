export PYTHONPATH="$PYTHONPATH:."

# sphere  10,20,40,80
# rosenbrock  10,20,40,80
# reb2-chain-weak  10,20,40,80
# reb2-chain-strong  10,20,40,80
# reb2-chain-alternating  10,20,40,80
# reb5-no-overlap  10,20,40,80
# reb5-small-overlap  9,21,41,81
# reb5-small-overlap-alternating  9,21,41,81
# osoreb  10,20,40,80
# reb5-large-overlap  10,20,40,80
# reb5-disjoint-pairs  9,18,36,72
# reb-grid  16,36,64,81

problem=$1
dimensionalities=$2
models="mp-hg-gbo-without_clique_seeding-conditional,mp-hg-gbo-with_clique_seeding-conditional,mp-hg-fb_no_order-without_clique_seeding-conditional,mp-hg-fb_no_order-with_clique_seeding-conditional,univariate,lt-fb-online-pruned,vkd-cma,full"

python rvgomea/cmd/run_set_of_bisections.py -o data/scalability-bisection-${problem} -p ${problem} -l ${models} -d ${dimensionalities} -r 5

echo "Done."
