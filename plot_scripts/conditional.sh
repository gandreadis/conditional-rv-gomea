export PYTHONPATH="$PYTHONPATH:."

python rvgomea/cmd/aggregate_set_of_bisections.py gecco-data/reb-chain-bisection-conditional

python rvgomea/plots/gecco/plot_bisection_matrices.py gecco-data/reb-chain-bisection-conditional uni-hg-gbo-without_clique_seeding-conditional,uni-hg-fb_no_order-without_clique_seeding-conditional,mp-hg-gbo-with_clique_seeding-conditional,mp-hg-fb_no_order-with_clique_seeding-conditional "(a) Static-UCond-HG,(b) FB-UCond-HG,(c) Static-MCond-HG-Max,(d) FB-MCond-HG-Max"
