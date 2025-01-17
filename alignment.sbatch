#!/usr/bin/bash
#SBATCH --job-name=alignment
#SBATCH --output=alignment.%j.out
#SBATCH --error=alignment.%j.err
#SBATCH --time=2-00:00:00
#SBATCH -p normal
#SBATCH -n 64
#SBATCH --mem=128GB	

export PATH=/home/groups/calvira/software/cellranger-9.0.0:$PATH
for dir in /scratch/users/cknutsen/vec/data/usftp21.novogene.com/01.RawData/*/
do
    dir2=${dir%*/}   
    dir2="${dir2##*/}"
    echo $dir2   
    cellranger count --id="${dir2}" \
                     --transcriptome=/home/groups/calvira/software/refdata-gex-GRCm39-2024-A \
                     --fastqs=$dir,/scratch/users/cknutsen/vec/data/partial_lane/usftp21.novogene.com/01.RawData/"${dir2}"/ \
                     --create-bam false \
                     --localcores 63 \
                     --localmem 124


done
