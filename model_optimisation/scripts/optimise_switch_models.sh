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

for NUM_LOOKAHEAD_EPMS in "${EPMS[@]}"; do
sbatch <<EOF
#!/bin/bash
#SBATCH -A p0026688
#SBATCH --job-name=switch_optimisation_lookahead_${NUM_LOOKAHEAD_EPMS}_normalized_afterwards
#SBATCH --output=${WORKDIR}/logs/switch_optimisation_lookahead_${NUM_LOOKAHEAD_EPMS}_normalized_afterwards.out
#SBATCH --error=${WORKDIR}/logs/switch_optimisation_lookahead_${NUM_LOOKAHEAD_EPMS}_normalized_afterwards.err
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