#!/bin/bash
# Submit interpolation-selector evaluation jobs for every dimension, metric,
# feature study, and lookahead count (0--20).
set -euo pipefail

ENV_PATH="/rwthfs/rz/cluster/home/p0027894/Dynamic_Selector/envs/dynamic_selector_venv"
WORKDIR="/hpcwork/p0027894/Dynamic_Selector/new_implementation"
DIMENSIONS=(5 10 20 40)
METRICS=(regret auc)

mkdir -p "$WORKDIR/logs"

FEATURE_STUDIES=(ela internal_state ela_internal_state)
INTERNAL_STATE_BUDGETS=$(seq 50 50 950)

for DIMENSION in "${DIMENSIONS[@]}"; do
  # tsfresh matrices do not depend on metric, feature study, or lookahead count.
  # Keep each budget in a separate job so extraction can run in parallel and a
  # memory failure does not discard the other cached matrices.
  PREPARE_JOB_IDS=()
  for BUDGET in $INTERNAL_STATE_BUDGETS; do
    PREPARE_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH -A p0027894
#SBATCH --job-name=interpolation_features_B${BUDGET}_D${DIMENSION}
#SBATCH --output=${WORKDIR}/logs/interpolation_features_B${BUDGET}_D${DIMENSION}_%j.out
#SBATCH --error=${WORKDIR}/logs/interpolation_features_B${BUDGET}_D${DIMENSION}_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=20G
#SBATCH --partition=c25ms
#SBATCH --cpus-per-task=12

set -euo pipefail
cd "$WORKDIR"
module load GCCcore/12.2.0
module load Python/3.10.8
source "$ENV_PATH/bin/activate"

PYTHONPATH="$WORKDIR/src" python -m interpolation_evaluate --dimension "$DIMENSION" --prepare-internal-state-features --internal-state-budget "$BUDGET"
EOF
)
    PREPARE_JOB_IDS+=("$PREPARE_JOB_ID")
  done
  PREPARE_DEPENDENCY="afterok:$(IFS=:; echo "${PREPARE_JOB_IDS[*]}")"

  for METRIC in "${METRICS[@]}"; do
    for FEATURE_STUDY in "${FEATURE_STUDIES[@]}"; do
      for LOOKAHEAD_COUNT in {0..20}; do
        DEPENDENCY_ARGS=()
        if [[ "$FEATURE_STUDY" != "ela" ]]; then
          DEPENDENCY_ARGS=(--dependency="$PREPARE_DEPENDENCY")
        fi
        sbatch "${DEPENDENCY_ARGS[@]}" <<EOF
#!/bin/bash
#SBATCH -A p0027894
#SBATCH --job-name=interpolation_eval_${FEATURE_STUDY}_L${LOOKAHEAD_COUNT}_D${DIMENSION}_${METRIC}
#SBATCH --output=${WORKDIR}/logs/interpolation_eval_${FEATURE_STUDY}_L${LOOKAHEAD_COUNT}_D${DIMENSION}_${METRIC}_%j.out
#SBATCH --error=${WORKDIR}/logs/interpolation_eval_${FEATURE_STUDY}_L${LOOKAHEAD_COUNT}_D${DIMENSION}_${METRIC}_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --partition=c25ms
#SBATCH --cpus-per-task=1

set -euo pipefail
cd "$WORKDIR"
module load GCCcore/12.2.0
module load Python/3.10.8
source "$ENV_PATH/bin/activate"

PYTHONPATH="$WORKDIR/src" python -m interpolation_evaluate --dimension "$DIMENSION" --metric "$METRIC" --feature-study "$FEATURE_STUDY" --lookahead-count "$LOOKAHEAD_COUNT"
EOF
      done
    done
  done

  for METRIC in "${METRICS[@]}"; do
    for FEATURE_STUDY in "${FEATURE_STUDIES[@]}"; do
      DEPENDENCY_ARGS=()
      if [[ "$FEATURE_STUDY" != "ela" ]]; then
        DEPENDENCY_ARGS=(--dependency="$PREPARE_DEPENDENCY")
      fi
      sbatch "${DEPENDENCY_ARGS[@]}" <<EOF
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
