#PBS -o ./output_reports
#PBS -e ./errors
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=11:00:00
#PBS -J 0-542
module load python/3.10.8
source ~/environments/my_env/bin/activate
cd /srv/scratch/z3545913/surf:summer
python3 -u test_run.py
