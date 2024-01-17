export PYTHONPATH="$PYTHONPATH:."

problem_index=$1

python rvgomea/cmd/run_extrapolated_pop_sizes.py gecco-data/scalability-bisection- gecco-data/scalability-extrapolated ${problem_index}
