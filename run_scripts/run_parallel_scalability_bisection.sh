export PYTHONPATH="$PYTHONPATH:."

job_array_index=$1

python rvgomea/cmd/run_parallel_set_of_bisections.py ${job_array_index}

echo "Done."
