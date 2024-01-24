export PYTHONPATH="$PYTHONPATH:."

problem_index=$1

if which python3.8 >/dev/null; then
    pycommand=python3.8
else
    pycommand=python
fi

$pycommand rvgomea/cmd/run_extrapolated_pop_sizes.py scalability-aggregated/scalability-bisection- scalability-extrapolated ${problem_index}
