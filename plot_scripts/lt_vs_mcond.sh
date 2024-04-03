export PYTHONPATH="$PYTHONPATH:."

python rvgomea/cmd/aggregate_set_of_bisections.py gecco-data/reb-chain-bisection-lt-vs-mcond

python rvgomea/plots/gecco/plot_bisection_reb_rotation_matrices.py gecco-data/reb-chain-bisection-lt-vs-mcond lt-fb-online-pruned,mp-hg-fb_no_order-without_clique_seeding-conditional,full "(a) FB-LT,(b) FB-MCond-HG,(c) Full"
