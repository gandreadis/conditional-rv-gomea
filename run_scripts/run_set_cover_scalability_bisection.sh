export PYTHONPATH="$PYTHONPATH:."

problem="reb-grid"
dimensionalities=16,36,64,81
# Full set:
#models="mp-hg-gbo-without_clique_seeding-conditional,mp-hg-gbo-with_clique_seeding-conditional,mp-hg-fb_no_order-without_clique_seeding-conditional,mp-hg-fb_no_order-with_clique_seeding-conditional,mp-fg-gbo-without_clique_seeding-non_conditional-set_cover,mp-hg-gbo-without_clique_seeding-conditional-set_cover"
models="mp-fg-gbo-without_clique_seeding-non_conditional-set_cover,mp-hg-gbo-without_clique_seeding-conditional-set_cover"

python rvgomea/cmd/run_set_of_bisections.py -o data/set-cover-bisection -p ${problem} -l ${models} -d ${dimensionalities} -r 5

echo "Done."
