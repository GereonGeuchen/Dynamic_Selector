#!/bin/bash

ENV_PATH="/home/p0027894/Dynamic_Selector/envs/dynamic_selector_venv"
WORKDIR="/hpcwork/p0027894/Dynamic_Selector/new_implementation"
PY_SCRIPT="selector.py"
set -euo pipefail

# With three arguments this retains the historical train-and-evaluate workflow.
# With four, it evaluates models trained on the third dataset against the fourth.
if [[ "$#" -ne 3 && "$#" -ne 4 ]]; then
  echo "Usage: $0 <dimension> <metric> <default|ma> [<evaluation-default|ma>]" >&2
  exit 1
fi

DIMENSION="$1"
METRIC="$2"
MODEL_DATASET="$3"
EVALUATION_DATASET="${4:-$MODEL_DATASET}"

case "$MODEL_DATASET" in
  default|ma) ;;
  *)
    echo "Usage: $0 <dimension> <metric> <default|ma> [<evaluation-default|ma>]" >&2
    exit 1
    ;;
esac

case "$EVALUATION_DATASET" in
  default|ma) ;;
  *)
    echo "Usage: $0 <dimension> <metric> <default|ma> [<evaluation-default|ma>]" >&2
    exit 1
    ;;
esac

if [[ "$#" -eq 3 ]]; then
  MODE="train-evaluate"
  DATASET_ARGS="--dataset $MODEL_DATASET --training-data-is-stored"
else
  MODE="evaluate"
  DATASET_ARGS="--dataset $EVALUATION_DATASET --model-dataset $MODEL_DATASET"
fi

mkdir -p "$WORKDIR/logs"

for lookahead_count in {0..20}; do
sbatch <<EOF
#!/bin/bash
#SBATCH -A p0026688
#SBATCH --job-name=selector_eval_${lookahead_count}_D${DIMENSION}_${METRIC}_${MODEL_DATASET}_on_${EVALUATION_DATASET}
#SBATCH --output=${WORKDIR}/logs/selector_eval_${lookahead_count}_D${DIMENSION}_${METRIC}_${MODEL_DATASET}_on_${EVALUATION_DATASET}.out
#SBATCH --error=${WORKDIR}/logs/selector_eval_${lookahead_count}_D${DIMENSION}_${METRIC}_${MODEL_DATASET}_on_${EVALUATION_DATASET}.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --partition=c25ms
#SBATCH --cpus-per-task=1

cd "$WORKDIR"
module load GCCcore/12.2.0
module load Python/3.10.8
source "$ENV_PATH/bin/activate"

python "$PY_SCRIPT" --mode $MODE --dimension "$DIMENSION" --metric "$METRIC" $DATASET_ARGS --lookahead-count "$lookahead_count"
EOF
done
