export PYTHONPATH="$PYTHONPATH:."

job_array_index=$1

if which python3.8 >/dev/null; then
    pycommand=python3.8
else
    pycommand=python
fi

$pycommand rvgomea/cmd/run_parallel_set_of_bisections.py ${job_array_index}

echo "Done."
