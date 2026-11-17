#!/bin/bash

ENV_PATH="/home/p0027894/Dynamic_Selector/envs/dynamic_selector_venv"
WORKDIR="/hpcwork/p0027894/Dynamic_Selector/new_implementation"
DIMENSION=40
FUNCTION_IDS="$(seq -s, 1 24)"

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
#SBATCH --job-name=data_collection_test_B${BUDGET}_${alg}_D${DIMENSION}
#SBATCH --output=${WORKDIR}/logs/data_collection_test_B${BUDGET}_${alg}_D${DIMENSION}.out
#SBATCH --error=${WORKDIR}/logs/data_collection_test_B${BUDGET}_${alg}_D${DIMENSION}.err
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --partition=c25ms
#SBATCH --cpus-per-task=1

cd "$WORKDIR"
module load GCCcore/12.2.0
module load Python/3.10.8
source "$ENV_PATH/bin/activate"

PYTHONPATH="$WORKDIR/src" python -m data_collection.data_collection "$BUDGET" "$alg" "$DIMENSION" "$FUNCTION_IDS"

EOF
  done
done

# alg="Non-elitist"
# BUDGET=1000
# sbatch <<EOF
# #!/bin/bash
# #SBATCH -A p0027894
# #SBATCH --job-name=data_collection_with_is_B${BUDGET}_${alg}_D${DIMENSION}  
# #SBATCH --output=${WORKDIR}/logs/data_collection_with_is_B${BUDGET}_${alg}_D${DIMENSION}.out
# #SBATCH --error=${WORKDIR}/logs/data_collection_with_is_B${BUDGET}_${alg}_D${DIMENSION}.err
# #SBATCH --time=24:00:00
# #SBATCH --mem=4G
# #SBATCH --partition=c25ms
# #SBATCH --cpus-per-task=1

# cd "$WORKDIR"
# # module load GCCcore/12.2.0
# # module load Python/3.10.8
# source "$ENV_PATH/bin/activate"

# python "$PY_SCRIPT" "$BUDGET" "$alg" "$DIMENSION"
# EOF
