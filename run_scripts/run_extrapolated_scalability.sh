export PYTHONPATH="$PYTHONPATH:."

problem_index=$1

python rvgomea/cmd/run_extrapolated_pop_sizes.py scalability-aggregated/scalability-bisection- scalability-extrapolated ${problem_index}
