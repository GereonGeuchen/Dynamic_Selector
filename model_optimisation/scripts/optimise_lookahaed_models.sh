#!/bin/bash

# Path to your unpacked conda-pack environment
ENV_PATH="$HOME/general-env"

# Project directory (where your Python code and data are)
WORKDIR="$HOME/Dokumente/Dynamic_Selector/model_optimisation/scripts"

# Path to your Python script
PY_SCRIPT="lookahead_optimisation.py"

# Create logs directory
mkdir -p "$WORKDIR/logs"

# # First sequence: 8 * [1..12]
# for i in $(seq 1 12); do
#   BUDGETS+=($((8 * i)))
# done

# # Second sequence: 50 * [1..20]
for i in $(seq 1 19); do
  BUDGETS+=($((50 * i)))
done
for BUDGET in "${BUDGETS[@]}"; do
    for I in $(seq 4 10); do
  sbatch <<EOF
#!/bin/bash
#SBATCH -A p0026688
#SBATCH --job-name=${BUDGET}_training_lookahead_t${I}
#SBATCH --output=${WORKDIR}/logs/${BUDGET}_training_lookahead_t${I}.out
#SBATCH --error=${WORKDIR}/logs/${BUDGET}_training_lookahead_t${I}.err
#SBATCH --time=10:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# Go to the working directory
cd $WORKDIR

# Activate your packed conda environment
source $ENV_PATH/bin/activate

# Run the Python tuning script
python $PY_SCRIPT $BUDGET $I
EOF
    done
done
