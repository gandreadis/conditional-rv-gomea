export PYTHONPATH="$PYTHONPATH:."

python rvgomea/cmd/run.py -l mp-hg-fb_no_order-with_clique_seeding-conditional -b -p reb2-chain-strong -d 20 -s 100 -i gecco-data/reb-chain-example/cond -r 3 -o -z
python rvgomea/cmd/run.py -l mp-fb-online-fg -b -p reb2-chain-strong -d 20 -s 200 -e 1e8 -i gecco-data/reb-chain-example/mp -r 1 -o -z
python rvgomea/cmd/run.py -l lt-fb-online-pruned -b -p reb2-chain-strong -d 20 -s 100 -i gecco-data/reb-chain-example/lt -r 1 -o -z
python rvgomea/cmd/run.py -l mp-fg-fb_no_order-with_clique_seeding-non_conditional -b -p reb2-chain-strong -d 20 -s 200 -e 1e8 -i gecco-data/reb-chain-example/omp -r 1 -o -z

python rvgomea/plots/gecco/plot_reb_chain_example.py gecco-data/reb-chain-example
