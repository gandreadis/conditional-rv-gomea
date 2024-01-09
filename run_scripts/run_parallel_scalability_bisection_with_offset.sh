export PYTHONPATH="$PYTHONPATH:."

job_array_index=$1
offset=$2

ind=$((job_array_index + offset))

if which python3.8 >/dev/null; then
    pycommand=python3.8
else
    pycommand=python
fi

$pycommand rvgomea/cmd/run_parallel_set_of_bisections.py ${ind}

echo "Done."
