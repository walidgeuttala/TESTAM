#!/bin/bash
#SBATCH -A c_gnn_001               # Account name to be debited
#SBATCH --job-name=gnn         # Job name
#SBATCH --time=0-12:00:00        # Maximum walltime (30 minutes)
#SBATCH --partition=gpu     # Select the ai partition
#SBATCH --gres=gpu:1          # Request 1 to 4 GPUs per node
#SBATCH --mem-per-cpu=80000       # Memory per CPU core (16 GB)
#SBATCH --nodes=1               # Request 1 node


python3 -u train.py --batch_size 64 --dropout 0.3 --seed -1 --save ./experiment/METR-LA_0/TESTAM --data ./data/METR-LA --adjdata ./data/METR-LA/adj_mx.pkl --device cuda --n_warmup_steps 4000 --warmup_epoch 0 --out_dim 1
 
# python3 model.py