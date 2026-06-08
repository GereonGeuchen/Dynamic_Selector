#!/bin/bash

ENV_PATH="$HOME/general-env"
WORKDIR="/home/p0026688/Dynamic_Selector/new_implementation"
PY_SCRIPT="data_collection.py"

mkdir -p "$WORKDIR/logs"

BUDGETS=()
for i in $(seq 1 20); do
  BUDGETS+=($((50 * i)))
done
algs_to_run=("BFGS" "MLSL" "Elitist" "Non-elitist" "DE" "PSO")

for BUDGET in "${BUDGETS[@]}"; do
  for alg in "${algs_to_run[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH -A p0026688
#SBATCH --job-name=data_collection_B${BUDGET}_${alg}
#SBATCH --output=${WORKDIR}/logs/data_collection_B${BUDGET}_${alg}.out
#SBATCH --error=${WORKDIR}/logs/data_collection_B${BUDGET}_${alg}.err
#SBATCH --time=96:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

cd "$WORKDIR"
source "$ENV_PATH/bin/activate"

python "$PY_SCRIPT" "$BUDGET" "$alg"
EOF
  done
done