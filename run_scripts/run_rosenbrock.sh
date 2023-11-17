#!/usr/bin/env sh

python rvgomea/cmd/run_set_of_bisections.py -o data/rosenbrock -p rosenbrock -l univariate,full,ucond-gg-gbo,ucond-fg-gbo,ucond-hg-gbo,mcond-hg-gbo -d 10,20,40,80 -r 5
