export PYTHONPATH="$PYTHONPATH:."

models=$1
id=$2
CONDITIONS=("1" "2" "3" "4" "5" "6")
ROTATIONS=("0" "5" "10" "15" "20" "25" "30" "35" "40" "45")

for c in "${CONDITIONS[@]}"
do
  echo "///// Condition number  ${c}"
  for r in "${ROTATIONS[@]}"
  do
    echo "///// Rotation angle  ${r}"
    python rvgomea/cmd/run_set_of_bisections.py -o data/reb-chain-bisection-${id} -p reb-chain-condition-${c}-rotation-${r} -l ${models} -d 20 -r 5
  done
done

echo "Done."
