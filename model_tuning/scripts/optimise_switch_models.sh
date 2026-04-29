#!/bin/bash

# Path to your unpacked conda-pack environment
ENV_PATH="$HOME/general-env"

# Project directory (where your Python code and data are)
WORKDIR="$HOME/Dokumente/Dynamic_Selector/model_optimisation/scripts"

# Path to your Python script
PY_SCRIPT="switch_model_optimisation.py"

for i in $(seq 0 19); do
  EPMS+=($((i)))
done

EPMS+=(-1)  # Add the case for no lookahead predictions as well

# EPMS=(-1)  # Add the case for no lookahead predictions as well

for NUM_LOOKAHEAD_EPMS in "${EPMS[@]}"; do
sbatch <<EOF
#!/bin/bash
#SBATCH -A p0026688
#SBATCH --job-name=switch_optimisation_lookahead_auc_${NUM_LOOKAHEAD_EPMS}_highest_new_norm
#SBATCH --output=${WORKDIR}/logs/switch_optimisation_lookahead_auc_${NUM_LOOKAHEAD_EPMS}_highest_new_norm.out
#SBATCH --error=${WORKDIR}/logs/switch_optimisation_lookahead_auc_${NUM_LOOKAHEAD_EPMS}_highest_new_norm.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=5

# Go to the working directory
cd $WORKDIR

# Activate your packed conda environment
source $ENV_PATH/bin/activate

# Run the Python tuning script
python $PY_SCRIPT $NUM_LOOKAHEAD_EPMS
EOF

done