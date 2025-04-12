#!/bin/bash
#PBS -N "cmake-build-singularity"
#PBS -l walltime=00:30:00
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=8gb:ngpus=1:scratch_local=64gb
#PBS -p 100
#PBS -j oe
#PBS -m e
set -e

test -n "${srcdir}" || { echo >&2 "Variable srcdir is not set."; exit 1; }

# parameters are:
# job name
# max walltime
# gpu queue
# number of machines, number of cpus, RAM and scratch size
# high priority
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

# Build.
cd "${SCRATCHDIR}"
# /usr/bin/singularity pull --force docker://floopcz/echo-state-networks:cuda-12.4
/usr/bin/singularity exec --nv --pwd "${WORKDIR}/${srcdir}" docker://floopcz/echo-state-networks:cuda-12.3 \
  /bin/bash -ce "
    rm -rvf build
    source /etc/profile
    cmake -GNinja -B build -DCMAKE_BUILD_TYPE=Release .
    # Warmup build to avoid timeout during link time.
    cmake --build build || true
    cmake --build build || true
    # Finally make sure the build finished successfully.
    cmake --build build
  "

# Copy the results from the scratch folder.
# cp -rv "${SCRATCHDIR}/*" "${WORKDIR}" || export CLEAN_SCRATCH=false
