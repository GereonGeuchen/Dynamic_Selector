#!/bin/bash

ENV_PATH="/home/p0027894/Dynamic_Selector/envs/dynamic_selector_venv"
WORKDIR="/hpcwork/p0027894/Dynamic_Selector/new_implementation"
PY_SCRIPT="data_collection.py"
DIMENSION=40
# Pass `true` as the first script argument to collect ManyAffine data:
# ./data_collection.sh true
USE_MA="${1:-true}"

case "$USE_MA" in
  true|false) ;;
  *)
    echo "Usage: $0 [true|false]" >&2
    exit 1
    ;;
esac

# `data_collection.py` expects use_ma as its seventh positional argument.
FUNCTION_IDS="$(seq -s, 1 24)"

mkdir -p "$WORKDIR/logs"

BUDGETS=()
for i in $(seq 1 19); do
  BUDGETS+=($((50 * i)))
done

algs_to_run=("BFGS" "MLSL" "Elitist" "DE" "PSO")
algs_to_run=("DE")


for BUDGET in "${BUDGETS[@]}"; do
  for alg in "${algs_to_run[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH -A p0027894
#SBATCH --job-name=data_collection_test_B${BUDGET}_${alg}_D${DIMENSION}_ma${USE_MA}
#SBATCH --output=${WORKDIR}/logs/data_collection_test_B${BUDGET}_${alg}_D${DIMENSION}_ma${USE_MA}.out
#SBATCH --error=${WORKDIR}/logs/data_collection_test_B${BUDGET}_${alg}_D${DIMENSION}_ma${USE_MA}.err
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --partition=c25ms
#SBATCH --cpus-per-task=1

cd "$WORKDIR"
module load GCCcore/12.2.0
module load Python/3.10.8
source "$ENV_PATH/bin/activate"

python "$PY_SCRIPT" "$BUDGET" "$alg" "$DIMENSION" "$FUNCTION_IDS" "" "$USE_MA"

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
