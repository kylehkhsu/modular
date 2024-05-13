#!/bin/bash
#SBATCH --partition=cpu
#SBTACH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --job-name=Optimiser			# Job name
#SBATCH --mem=32gb				# Job Memory Request
#SBATCH --time=24:00:00				# Time limit hrs:min:sec
#SBATCH --output=./logs/serial_test_%j.log	# Standard output file
#SBATCH --error=./logs/error_%j.err		# Standard error file
#SBATCH --mail-type=ALL				# When to send email notifications
#SBATCH --mail-user=dorrellwec@gmail.com	# Where to send them
#SBATCH --array=1-5


pwd; hostname; date

python Xie_2022_norepeats.py
