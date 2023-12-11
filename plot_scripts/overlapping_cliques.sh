export PYTHONPATH="$PYTHONPATH:."

python rvgomea/cmd/aggregate_set_of_bisections.py gecco-data/reb-chain-bisection-overlapping-cliques

python rvgomea/plots/gecco/plot_bisection_reb_rotation_matrices.py gecco-data/reb-chain-bisection-overlapping-cliques mp-fb-online-fg,mp-fg-fb_no_order-with_clique_seeding-non_conditional,full "(a) FB-MP,(b) FB-MP-MC,(c) Full"
