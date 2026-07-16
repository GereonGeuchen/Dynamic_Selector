#!/bin/bash

ENV_PATH="$HOME/general-env"
WORKDIR="/home/p0026688/Dynamic_Selector/new_implementation"
PY_SCRIPT="selector.py"

set -euo pipefail

mkdir -p "$WORKDIR/logs"

sbatch <<EOF
#!/bin/bash
#SBATCH -A p0027894
#SBATCH --job-name=selector
#SBATCH --output=${WORKDIR}/logs/selector.out
#SBATCH --error=${WORKDIR}/logs/selector.err
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

cd "$WORKDIR"
source "$ENV_PATH/bin/activate"

python "$PY_SCRIPT" --mode build-switch-data
EOF