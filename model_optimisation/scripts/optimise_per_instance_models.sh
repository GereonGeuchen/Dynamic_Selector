#!/bin/bash

# Path to your unpacked conda-pack environment
ENV_PATH="$HOME/general-env"

# Project directory (where your Python code and data are)
WORKDIR="$HOME/Dokumente/Dynamic_Selector/model_optimisation/scripts"

# Path to your Python script
PY_SCRIPT="tune_per_instance_selector.py"
sbatch <<EOF
#!/bin/bash
#SBATCH -A p0026688
#SBATCH --job-name=per_instance_selector_optimisation
#SBATCH --output=${WORKDIR}/logs/per_instance_selector_optimisation.out
#SBATCH --error=${WORKDIR}/logs/per_instance_selector_optimisation.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=5

# Go to the working directory
cd $WORKDIR

# Activate your packed conda environment
source $ENV_PATH/bin/activate

# Run the Python tuning script
python $PY_SCRIPT $BUDGET
EOF