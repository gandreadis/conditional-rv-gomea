export PYTHONPATH="$PYTHONPATH:."

python rvgomea/cmd/run.py -l mp-hg-fb_no_order-with_clique_seeding-conditional -p reb-grid -d 9 -s 100 -i gecco-data/reb-grid-example/cond -r 1 -o -z
python rvgomea/cmd/run.py -l mp-fb-online-fg -p reb-grid -d 9 -s 100 -i gecco-data/reb-grid-example/mp -r 1 -o -z
python rvgomea/cmd/run.py -l lt-fb-online-pruned -p reb-grid -d 9 -s 100 -i gecco-data/reb-grid-example/lt -r 1 -o -z
python rvgomea/cmd/run.py -l mp-fg-fb_no_order-with_clique_seeding-non_conditional -p reb-grid -d 9 -s 200 -i gecco-data/reb-grid-example/omp -r 1 -o -z

python rvgomea/plots/gecco/plot_reb_grid_example.py gecco-data/reb-grid-example
