#!/bin/bash

ENV_PATH="$HOME/general-env"
WORKDIR="/home/p0026688/Dynamic_Selector/new_implementation"
PY_SCRIPT="data_collection.py"
DIMENSION=40

mkdir -p "$WORKDIR/logs"

BUDGETS=()
for i in $(seq 1 19); do
  BUDGETS+=($((50 * i)))
done

algs_to_run=("BFGS" "MLSL" "Elitist" "DE" "PSO")

for BUDGET in "${BUDGETS[@]}"; do
  for alg in "${algs_to_run[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH -A p0027894
#SBATCH --job-name=data_collection_B${BUDGET}_${alg}
#SBATCH --output=${WORKDIR}/logs/data_collection_B${BUDGET}_${alg}.out
#SBATCH --error=${WORKDIR}/logs/data_collection_B${BUDGET}_${alg}.err
#SBATCH --time=48:00:00
#SBATCH --mem=4G
#SBATCH --partition=c23ms
#SBATCH --cpus-per-task=1

cd "$WORKDIR"
source "$ENV_PATH/bin/activate"

python "$PY_SCRIPT" "$BUDGET" "$alg" "$DIMENSION"
EOF

  done
done

alg="Non-elitist"
BUDGET=1000
sbatch <<EOF
#!/bin/bash
#SBATCH -A p0027894
#SBATCH --job-name=data_collection_B${BUDGET}_${alg}
#SBATCH --output=${WORKDIR}/logs/data_collection_B${BUDGET}_${alg}.out
#SBATCH --error=${WORKDIR}/logs/data_collection_B${BUDGET}_${alg}.err
#SBATCH --time=48:00:00
#SBATCH --mem=4G
#SBATCH --partition=c23ms
#SBATCH --cpus-per-task=1

cd "$WORKDIR"
source "$ENV_PATH/bin/activate"

python "$PY_SCRIPT" "$BUDGET" "$alg" "$DIMENSION"
EOF
