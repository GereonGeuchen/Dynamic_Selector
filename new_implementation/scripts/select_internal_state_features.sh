#!/bin/bash
# Submit tsfresh/Random-Forest feature-selection jobs for budgets 50, 100, ..., 950.
# Usage: bash select_internal_state_features.sh [DIMENSION]

set -euo pipefail

ENV_PATH="/home/p0027894/Dynamic_Selector/envs/dynamic_selector_venv"
WORKDIR="/hpcwork/p0027894/Dynamic_Selector/new_implementation"
DIMENSION="${1:-40}"
CPUS=30
INPUT_CSV="data/dim_${DIMENSION}_with_internal_state/internal_state/internal_state_Non-elitist_B1000_${DIMENSION}D.csv"

mkdir -p "$WORKDIR/logs"

for BUDGET in $(seq 50 50 950); do
    sbatch <<EOF
#!/bin/bash
#SBATCH -A p0027894
#SBATCH --job-name=tsfresh_fid_B${BUDGET}_${DIMENSION}D
#SBATCH --output=${WORKDIR}/logs/tsfresh_fid_B${BUDGET}_${DIMENSION}D_%j.out
#SBATCH --error=${WORKDIR}/logs/tsfresh_fid_B${BUDGET}_${DIMENSION}D_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --partition=c25ms
#SBATCH --cpus-per-task=${CPUS}

set -euo pipefail
cd "$WORKDIR"
module load GCCcore/12.2.0
module load Python/3.10.8
source "$ENV_PATH/bin/activate"

python select_internal_state_features.py "$INPUT_CSV" \\
    --dimension "$DIMENSION" \\
    --budget "$BUDGET" \\
    --n-jobs "$CPUS" \\
    --save-series
EOF
done
