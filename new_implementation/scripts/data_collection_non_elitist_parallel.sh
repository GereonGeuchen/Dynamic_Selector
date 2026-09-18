#!/bin/bash
# Submit 24 independent Non-elitist/B1000 jobs (one per BBOB function) and a
# merge job which runs only when every array element succeeds.
set -euo pipefail

ENV_PATH="/rwthfs/rz/cluster/home/p0027894/Dynamic_Selector/envs/dynamic_selector_venv"
WORKDIR="/hpcwork/p0027894/Dynamic_Selector/new_implementation"
BUDGET=1000
DIMENSION=40
ALGORITHM="Non-elitist"
# Pass `true` as the first argument to run the array on ManyAffine functions:
# ./data_collection_non_elitist_parallel.sh true
USE_MA="${1:-true}"

case "$USE_MA" in
  true) MERGE_MA_FLAG="--use-ma" ;;
  false) MERGE_MA_FLAG="" ;;
  *)
    echo "Usage: $0 [true|false]" >&2
    exit 1
    ;;
esac

mkdir -p "$WORKDIR/logs"

# Do not remove part files while an earlier array is running.  A fresh run may
# safely overwrite its final files during the merge step.
ARRAY_JOB_ID=$(sbatch --parsable --array=1-24 <<EOF
#!/bin/bash
#SBATCH -A p0027894
#SBATCH --job-name=ela_nonelitist_with_state_B${BUDGET}_ma${USE_MA}
#SBATCH --output=${WORKDIR}/logs/ela_nonelitist_with_state_B${BUDGET}_%A_%a_ma${USE_MA}.out
#SBATCH --error=${WORKDIR}/logs/ela_nonelitist_with_state_B${BUDGET}_%A_%a_ma${USE_MA}.err
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --partition=c25ms
#SBATCH --cpus-per-task=1

cd "$WORKDIR"
module load GCCcore/12.2.0
module load Python/3.10.8
source "$ENV_PATH/bin/activate"
python data_collection.py "$BUDGET" "$ALGORITHM" "$DIMENSION" "\$SLURM_ARRAY_TASK_ID" ".part-\$SLURM_ARRAY_TASK_ID" "$USE_MA"
EOF
)

MERGE_JOB_ID=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB_ID} <<EOF
#!/bin/bash
#SBATCH -A p0027894
#SBATCH --job-name=merge_ela_nonelitist_with_state_B${BUDGET}_ma${USE_MA}
#SBATCH --output=${WORKDIR}/logs/merge_ela_nonelitist_with_state_B${BUDGET}_%j.out
#SBATCH --error=${WORKDIR}/logs/merge_ela_nonelitist_with_state_B${BUDGET}_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --partition=c25ms
#SBATCH --cpus-per-task=1

cd "$WORKDIR"

module load GCCcore/12.2.0
module load Python/3.10.8

source "$ENV_PATH/bin/activate"
python merge_data_collection_parts.py "$BUDGET" "$ALGORITHM" "$DIMENSION" $MERGE_MA_FLAG
EOF
)

echo "Submitted array job ${ARRAY_JOB_ID}; merge job ${MERGE_JOB_ID}."
