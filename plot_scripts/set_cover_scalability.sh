export PYTHONPATH="$PYTHONPATH:."

python rvgomea/cmd/aggregate_set_of_bisections.py gecco-data/scalability-bisection-reb-grid

python rvgomea/plots/gecco/plot_bisection_scalability.py gecco-data/scalability-bisection-reb-grid reb-grid "-1" mp-hg-gbo-without_clique_seeding-conditional,mp-hg-gbo-with_clique_seeding-conditional,mp-hg-fb_no_order-without_clique_seeding-conditional,mp-hg-fb_no_order-with_clique_seeding-conditional,mp-fg-gbo-without_clique_seeding-non_conditional-set_cover,mp-hg-gbo-without_clique_seeding-conditional-set_cover
