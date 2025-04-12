#!/bin/bash
#PBS -N "run-cuda-experiment"
#PBS -l walltime=24:00:00
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l select=1:ncpus=1:mem=2gb:ngpus=1:gpu_mem=24gb:scratch_local=16gb:cuda_version=12.4
#PBS -p 10
#PBS -j oe
#PBS -m a
set -e

test -n "${cmd}" || { echo >&2 "Variable cmd is not set."; exit 1; }

# parameters are:
# job name
# max walltime
# gpu queue
# number of machines, number of cpus, RAM and scratch size
# positive priority to be able to create new preferred jobs
# pipe error output to standard output
# send e-mail on job failure

trap 'clean_scratch' TERM EXIT

# Set up used folders and print debug info.
test -n "${PBS_O_WORKDIR}" || { echo >&2 "Variable PBS_O_WORKDIR is not set!"; exit 1; }
test -n "${PBS_O_HOME}" || { echo >&2 "Variable PBS_O_HOME is not set!"; exit 1; }
test -n "${SCRATCHDIR}" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }
WORKDIR="$(realpath "${PBS_O_WORKDIR}")"
HOMEDIR="$(realpath "${PBS_O_HOME}")"
echo "${PBS_JOBID} is running on node `hostname -f` in a scratch directory ${SCRATCHDIR}" | tee -a "${HOMEDIR}/jobs_info.txt"
echo "workdir is ${WORKDIR}, homedir is ${HOMEDIR}"
echo "date is $(date)"
cd "${SCRATCHDIR}"

# Set up Singularity env variables.
export SINGULARITY_CACHEDIR="${HOMEDIR}/.singularity"
mkdir -p "${SINGULARITY_CACHEDIR}"
export SINGULARITY_LOCALCACHEDIR="${SCRATCHDIR}/singularity"
mkdir -p "${SINGULARITY_LOCALCACHEDIR}"
export SINGULARITY_TMPDIR="${SCRATCHDIR}/singularity"
mkdir -p "${SINGULARITY_TMPDIR}"
echo "Singularity cache dir is ${SINGULARITY_CACHEDIR}."
/usr/bin/singularity cache list

# Run.
export OMP_NUM_THREADS="${PBS_NUM_PPN}"
/usr/bin/singularity exec --nv --pwd "${WORKDIR}" docker://floopcz/echo-state-networks:cuda-12.3 ${cmd}

# Copy the results from the scratch folder.
if [ "$(ls -A "${SCRATCHDIR}")" ]; then
    cp -rv "${SCRATCHDIR}"/* "${WORKDIR}" || export CLEAN_SCRATCH=false
fi
