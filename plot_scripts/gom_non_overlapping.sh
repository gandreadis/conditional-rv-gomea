export PYTHONPATH="$PYTHONPATH:."

python rvgomea/cmd/aggregate_set_of_bisections.py gecco-data/reb-chain-bisection-gom-non-overlapping

python rvgomea/plots/gecco/plot_bisection_matrices.py gecco-data/reb-chain-bisection-gom-non-overlapping mp-fb-online-fg,mp-fb-online-hg "(a) FB-MP-FG,(b) FB-MP-HG"
