#!/bin/bash
# Submit all collection jobs for the fixed 20 affine interpolation experiments.
#
# The B50--B950 jobs are independent, one per switching budget.  The B1000
# Non-elitist run computes ELA features and is therefore submitted once per
# interpolation ID, followed by a dependent merge job.
#
# Usage: bash scripts/interpolation_data_collection.sh [dimension] [repetitions]
set -euo pipefail

ENV_PATH="/rwthfs/rz/cluster/home/p0027894/Dynamic_Selector/envs/dynamic_selector_venv"
WORKDIR="/hpcwork/p0027894/Dynamic_Selector/new_implementation"
DIMENSION="${1:-40}"
REPETITIONS="${2:-20}"
INTERPOLATION_IDS=($(seq 1 20))

mkdir -p "$WORKDIR/logs"

# At budgets below 1000, the collector automatically runs the five A2
# algorithms and skips Non-elitist (whose trajectory is the B1000 run).
for BUDGET in $(seq 50 50 950); do
  sbatch <<EOF
#!/bin/bash
#SBATCH -A p0027894
#SBATCH --job-name=interpolation_B${BUDGET}_D${DIMENSION}
#SBATCH --output=${WORKDIR}/logs/interpolation_B${BUDGET}_D${DIMENSION}_%j.out
#SBATCH --error=${WORKDIR}/logs/interpolation_B${BUDGET}_D${DIMENSION}_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --partition=c25ms
#SBATCH --cpus-per-task=1

set -euo pipefail
cd "$WORKDIR"
module load GCCcore/12.2.0
module load Python/3.10.8
source "$ENV_PATH/bin/activate"
python interpolation_data_collection.py "$BUDGET" --dimension "$DIMENSION" --repetitions "$REPETITIONS"
EOF
done

ELA_JOB_IDS=()
for INTERPOLATION_ID in "${INTERPOLATION_IDS[@]}"; do
  ELA_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH -A p0027894
#SBATCH --job-name=interpolation_ela_B1000_I${INTERPOLATION_ID}_D${DIMENSION}
#SBATCH --output=${WORKDIR}/logs/interpolation_ela_B1000_I${INTERPOLATION_ID}_D${DIMENSION}_%j.out
#SBATCH --error=${WORKDIR}/logs/interpolation_ela_B1000_I${INTERPOLATION_ID}_D${DIMENSION}_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --partition=c25ms
#SBATCH --cpus-per-task=1

set -euo pipefail
cd "$WORKDIR"
module load GCCcore/12.2.0
module load Python/3.10.8
source "$ENV_PATH/bin/activate"
python interpolation_data_collection.py 1000 --dimension "$DIMENSION" --algorithms Non-elitist --repetitions "$REPETITIONS" --interpolation-ids "$INTERPOLATION_ID" --suffix ".part-$INTERPOLATION_ID"
EOF
)
  ELA_JOB_IDS+=("$ELA_JOB_ID")
done

PART_SUFFIXES=$(printf '.part-%s,' "${INTERPOLATION_IDS[@]}")
PART_SUFFIXES=${PART_SUFFIXES%,}
DEPENDENCY_IDS=$(IFS=:; printf '%s' "${ELA_JOB_IDS[*]}")
MERGE_JOB_ID=$(sbatch --parsable --dependency=afterok:${DEPENDENCY_IDS} <<EOF
#!/bin/bash
#SBATCH -A p0027894
#SBATCH --job-name=merge_interpolation_ela_B1000_D${DIMENSION}
#SBATCH --output=${WORKDIR}/logs/merge_interpolation_ela_B1000_D${DIMENSION}_%j.out
#SBATCH --error=${WORKDIR}/logs/merge_interpolation_ela_B1000_D${DIMENSION}_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --partition=c25ms
#SBATCH --cpus-per-task=1

set -euo pipefail
cd "$WORKDIR"
module load GCCcore/12.2.0
module load Python/3.10.8
source "$ENV_PATH/bin/activate"
python interpolation_data_collection.py 1000 --dimension "$DIMENSION" --algorithms Non-elitist --merge-suffixes "$PART_SUFFIXES"
EOF
)

echo "Submitted B50--B950 jobs plus ${#ELA_JOB_IDS[@]} B1000 ELA jobs; merge job ${MERGE_JOB_ID}."
