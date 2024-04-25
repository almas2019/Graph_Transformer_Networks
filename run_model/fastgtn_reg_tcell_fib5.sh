#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1        # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=31500M        # Memory proportional to GPUs: 31500 Cedar, 63500 Graham.
#SBATCH -J reg_tcellfib5
#SBATCH -N 1
# ---------------------------------------------------------------------
echo "Current working directory: $(pwd)"
echo "Starting run at: $(date)"
# ---------------------------------------------------------------------
echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""
# ---------------------------------------------------------------------
# activate your virtual environment
module load python/3.10
module load scipy-stack
source /project/def-gregorys/almas/spgraph_env/bin/activate
nvidia-smi


cd /home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks
python main.py --dataset graph_het_subsampled_tcell_fib5 --model FastGTN --num_layers 2 --epoch 100 --lr 0.01 --channel_agg mean --num_channels 2 --runs 1 --save_metrics
# number of runs --> how many time they initialize model and run all 100 epochs again (repeat the process)
# Each run involves initializing the model from scratch, training it for the specified number of epochs, and evaluating its performance.
# represents the number of times the entire dataset is passed forward and backward through the neural network during training.

