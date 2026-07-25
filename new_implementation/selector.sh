#!/bin/bash

ENV_PATH="$HOME/general-env"
WORKDIR="/home/p0026688/Dynamic_Selector/new_implementation"
PY_SCRIPT="selector.py"

set -euo pipefail

mkdir -p "$WORKDIR/logs"

for LOOKAHEAD_COUNT in {12..15}; do
sbatch <<EOF
#!/bin/bash
#SBATCH -A p0027894
#SBATCH --job-name=selector_lookahead_regret_${LOOKAHEAD_COUNT}
#SBATCH --output=${WORKDIR}/logs/selector_lookahead_regret_${LOOKAHEAD_COUNT}.out
#SBATCH --error=${WORKDIR}/logs/selector_lookahead_regret_${LOOKAHEAD_COUNT}.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

cd "$WORKDIR"
source "$ENV_PATH/bin/activate"

python "$PY_SCRIPT" --mode evaluate --lookahead-count "$LOOKAHEAD_COUNT"
EOF
done
