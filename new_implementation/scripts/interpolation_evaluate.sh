#!/bin/bash
# Submit one interpolation-selector evaluation job per lookahead count (0--20).
#
# Usage: bash scripts/interpolation_evaluate.sh [dimension] [metric]
set -euo pipefail

ENV_PATH="/rwthfs/rz/cluster/home/p0027894/Dynamic_Selector/envs/dynamic_selector_venv"
WORKDIR="/hpcwork/p0027894/Dynamic_Selector/new_implementation"
DIMENSION="${1:-40}"
METRIC="${2:-regret}"

case "$METRIC" in
  regret|auc) ;;
  *)
    echo "Usage: $0 [dimension] [regret|auc]" >&2
    exit 1
    ;;
esac

mkdir -p "$WORKDIR/logs"

for LOOKAHEAD_COUNT in {0..20}; do
  sbatch <<EOF
#!/bin/bash
#SBATCH -A p0027894
#SBATCH --job-name=interpolation_eval_L${LOOKAHEAD_COUNT}_D${DIMENSION}_${METRIC}
#SBATCH --output=${WORKDIR}/logs/interpolation_eval_L${LOOKAHEAD_COUNT}_D${DIMENSION}_${METRIC}_%j.out
#SBATCH --error=${WORKDIR}/logs/interpolation_eval_L${LOOKAHEAD_COUNT}_D${DIMENSION}_${METRIC}_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --partition=c25ms
#SBATCH --cpus-per-task=1

set -euo pipefail
cd "$WORKDIR"
module load GCCcore/12.2.0
module load Python/3.10.8
source "$ENV_PATH/bin/activate"

python interpolation_evaluate.py --dimension "$DIMENSION" --metric "$METRIC" --lookahead-count "$LOOKAHEAD_COUNT"
EOF
done
