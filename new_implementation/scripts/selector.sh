#!/bin/bash

ENV_PATH="/home/p0027894/Dynamic_Selector/envs/dynamic_selector_venv"
WORKDIR="/hpcwork/p0027894/Dynamic_Selector/new_implementation"
set -euo pipefail

DIMENSIONS=(5 10 20 40)
METRICS=(regret auc)
FEATURE_STUDIES=(ela internal_state ela_internal_state)
DATASET="default"


mkdir -p "$WORKDIR/logs"

for DIMENSION in "${DIMENSIONS[@]}"; do
    for METRIC in "${METRICS[@]}"; do
        for FEATURE_STUDY in "${FEATURE_STUDIES[@]}"; do
            # `train` creates the selection, lookahead, and switching training
            # tables before fitting and storing all model variants.
            TRAIN_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH -A p0027894
#SBATCH --job-name=selector_train_D${DIMENSION}_${METRIC}_${FEATURE_STUDY}_${DATASET}
#SBATCH --output=${WORKDIR}/logs/selector_train_D${DIMENSION}_${METRIC}_${FEATURE_STUDY}_${DATASET}.out
#SBATCH --error=${WORKDIR}/logs/selector_train_D${DIMENSION}_${METRIC}_${FEATURE_STUDY}_${DATASET}.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --partition=c25ms
#SBATCH --cpus-per-task=8

cd "$WORKDIR"
module load GCCcore/12.2.0
module load Python/3.10.8
source "$ENV_PATH/bin/activate"

PYTHONPATH="$WORKDIR/src" python "$WORKDIR/src/selector.py" --mode train --dimension "$DIMENSION" --metric "$METRIC" --dataset "$DATASET" --feature-study "$FEATURE_STUDY"
EOF
)

            for LOOKAHEAD_COUNT in $(seq 0 20); do
                sbatch --dependency="afterok:${TRAIN_JOB_ID}" <<EOF
#!/bin/bash
#SBATCH -A p0027894
#SBATCH --job-name=selector_eval_D${DIMENSION}_${METRIC}_${FEATURE_STUDY}_${DATASET}_L${LOOKAHEAD_COUNT}
#SBATCH --output=${WORKDIR}/logs/selector_eval_D${DIMENSION}_${METRIC}_${FEATURE_STUDY}_${DATASET}_L${LOOKAHEAD_COUNT}.out
#SBATCH --error=${WORKDIR}/logs/selector_eval_D${DIMENSION}_${METRIC}_${FEATURE_STUDY}_${DATASET}_L${LOOKAHEAD_COUNT}.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --partition=c25ms
#SBATCH --cpus-per-task=1

cd "$WORKDIR"
module load GCCcore/12.2.0
module load Python/3.10.8
source "$ENV_PATH/bin/activate"

PYTHONPATH="$WORKDIR/src" python "$WORKDIR/src/selector.py" --mode evaluate --dimension "$DIMENSION" --metric "$METRIC" --dataset "$DATASET" --feature-study "$FEATURE_STUDY" --lookahead-count "$LOOKAHEAD_COUNT"
EOF
            done
        done
done
done
