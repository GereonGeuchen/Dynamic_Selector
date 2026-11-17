#!/bin/bash
# Evaluate standard-BBOB static selectors on interpolation functions.
#
# This assumes the interpolation tsfresh matrices already exist. The static
# models must already have been trained on the standard BBOB data.
set -euo pipefail

ENV_PATH="/rwthfs/rz/cluster/home/p0027894/Dynamic_Selector/envs/dynamic_selector_venv"
WORKDIR="/hpcwork/p0027894/Dynamic_Selector/new_implementation"
# This interpolation study is intentionally limited to 5D and compares pure
# ELA predictors with the combined ELA + internal-state predictor set.
DIMENSIONS=(5)
METRICS=(regret auc)
FEATURE_STUDIES=(ela ela_internal_state)

mkdir -p "$WORKDIR/logs"

for DIMENSION in "${DIMENSIONS[@]}"; do
  for METRIC in "${METRICS[@]}"; do
    for FEATURE_STUDY in "${FEATURE_STUDIES[@]}"; do
      sbatch <<EOF
#!/bin/bash
#SBATCH -A p0027894
#SBATCH --job-name=interpolation_static_eval_D${DIMENSION}_${METRIC}_${FEATURE_STUDY}
#SBATCH --output=${WORKDIR}/logs/interpolation_static_eval_D${DIMENSION}_${METRIC}_${FEATURE_STUDY}_%j.out
#SBATCH --error=${WORKDIR}/logs/interpolation_static_eval_D${DIMENSION}_${METRIC}_${FEATURE_STUDY}_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --partition=c25ms
#SBATCH --cpus-per-task=8

set -euo pipefail
cd "$WORKDIR"
module load GCCcore/12.2.0
module load Python/3.10.8
source "$ENV_PATH/bin/activate"

PYTHONPATH="$WORKDIR/src" python "$WORKDIR/src/static_selection_model.py" --mode evaluate --interpolation --dimension "$DIMENSION" --metric "$METRIC" --feature-study "$FEATURE_STUDY"
EOF
    done
  done
done
