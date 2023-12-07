export PYTHONPATH="$PYTHONPATH:."

python rvgomea/cmd/aggregate_set_of_bisections.py gecco-data/reb-chain-bisection-gom-overlapping

python rvgomea/plots/gecco/plot_bisection_matrices.py gecco-data/reb-chain-bisection-gom-overlapping mp-fg-fb_no_order-with_clique_seeding-non_conditional,mp-hg-fb_no_order-with_clique_seeding-non_conditional "(a) FB-MP-FG,(b) FB-MP-HG"
