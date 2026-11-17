#!/bin/bash
# Submit six-choice static-selector training and evaluation jobs.
# Edit the arrays below to restrict the experiment grid.
set -euo pipefail

ENV_PATH="/home/p0027894/Dynamic_Selector/envs/dynamic_selector_venv"
WORKDIR="/hpcwork/p0027894/Dynamic_Selector/new_implementation"
DIMENSIONS=(5 10 20 40)
METRICS=(regret auc)
FEATURE_STUDIES=(ela internal_state ela_internal_state)

mkdir -p "$WORKDIR/logs"

for DIMENSION in "${DIMENSIONS[@]}"; do
    for METRIC in "${METRICS[@]}"; do
        for FEATURE_STUDY in "${FEATURE_STUDIES[@]}"; do
            TRAIN_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH -A p0027894
#SBATCH --job-name=static_selector_train_D${DIMENSION}_${METRIC}_${FEATURE_STUDY}
#SBATCH --output=${WORKDIR}/logs/static_selector_train_D${DIMENSION}_${METRIC}_${FEATURE_STUDY}_%j.out
#SBATCH --error=${WORKDIR}/logs/static_selector_train_D${DIMENSION}_${METRIC}_${FEATURE_STUDY}_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --partition=c23ms
#SBATCH --cpus-per-task=8

set -euo pipefail
cd "$WORKDIR"
module load GCCcore/12.2.0
module load Python/3.10.8
source "$ENV_PATH/bin/activate"
PYTHONPATH="$WORKDIR/src" python "$WORKDIR/src/static_selection_model.py" --mode train --dimension "$DIMENSION" --metric "$METRIC" --feature-study "$FEATURE_STUDY"
EOF
)

            sbatch --dependency="afterok:${TRAIN_JOB_ID}" <<EOF
#!/bin/bash
#SBATCH -A p0027894
#SBATCH --job-name=static_selector_eval_D${DIMENSION}_${METRIC}_${FEATURE_STUDY}
#SBATCH --output=${WORKDIR}/logs/static_selector_eval_D${DIMENSION}_${METRIC}_${FEATURE_STUDY}_%j.out
#SBATCH --error=${WORKDIR}/logs/static_selector_eval_D${DIMENSION}_${METRIC}_${FEATURE_STUDY}_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --partition=c23ms
#SBATCH --cpus-per-task=8

set -euo pipefail
cd "$WORKDIR"
module load GCCcore/12.2.0
module load Python/3.10.8
source "$ENV_PATH/bin/activate"
PYTHONPATH="$WORKDIR/src" python "$WORKDIR/src/static_selection_model.py" --mode evaluate --dimension "$DIMENSION" --metric "$METRIC" --feature-study "$FEATURE_STUDY"
EOF
        done
    done
done
