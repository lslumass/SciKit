#!/bin/bash
#SBATCH -c 40                                     # Number of Cores per Task
#SBATCH --mem=64GB                                # Requested Memory
#SBATCH -t 2-00:00:00
#SBATCH -p cpu-preempt,cpu                                    # Partition
#SBATCH -o slurm-ana-%j.out  # %j = job ID
#SBATCH --job-name=analysis

#scical rg --top conf.psf --traj system.dcd --start 1000 --nproc 40
#scical saxs --top conf.psf --traj system.dcd --start 10000 --stride 10 --nproc 40
#scical dssp --top conf.psf --traj system.dcd --start 1000 --nproc 40
scical contacts --top conf.psf --traj system.dcd --components "A:1" --start 1000 --stride 10 --nproc 40
