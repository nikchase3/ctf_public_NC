# This script creates a PBS file that runs one hyperameter setting
# on a single node.
import os

# generates a file called 'run.pbs' in the directory this python file is in
pbs_fn = 'run.pbs'
curr_dir = os.getcwd()
pbs_path = os.path.join(curr_dir, pbs_fn)

trainingFilename = 'test_job.py'
walltime = '00:30:00'
num_nodes = '1'
cores_per_node = '16'
gpu = 'P100'
jobname ='my_training_job_name'
netid = 'jheglun2'

directory = curr_dir.replace('\\', '/')

with open(pbs_path, 'w') as f:
    f.write("#!/bin/bash\n")

    f.write("#PBS -N {}\n".format(jobname))
    f.write("#PBS -l walltime={}\n".format(walltime))
    f.write("#PBS -l nodes={}:ppn={}:{}\n".format(num_nodes, cores_per_node, gpu))
    f.write("#PBS -j oe\n")
    f.write("#PBS -M {}@illinois.edu\n".format(netid))
    f.write("#PBS -m be\n")

    f.write('module load python/3\n')
    f.write('module load anaconda/3\n')
    f.write('module load cuda/9.2\n')
    f.write('source activate py37\n')
    f.write("cd $PBS_O_WORKDIR\n")
    f.write("python {}\n".format(trainingFilename))

