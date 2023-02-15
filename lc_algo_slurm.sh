#!/bin/bash
#SBATCH --qos=training
#SBATCH --gres=gpu:1
#SBATCH --job-name=lc-alexnet
#SBATCH --output=/home/%u/slurmLogs/lc_compression_%a_%A.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --array=8

ALPHA_OPTS=(1e-10 1e-9 1e-8 1e-7 1e-10 1e-9 1e-8 1e-7 1e-10 1e-9 1e-8 1e-7 1e-10 1e-9 1e-8 1e-7 1e-10 1e-9 1e-8 1e-7)
MU_INIT_OPTS=(2e-4 2e-4 2e-4 2e-4 2e-3 2e-3 2e-3 2e-3 2e-2 2e-2 2e-2 2e-2 2e-1 2e-1 2e-1 2e-1)

if [ "${SLURM_ARRAY_TASK_ID}" == "" ]; then
    SLURM_ARRAY_TASK_ID=$1
    EXTRA_PARS=${@:2:1}
else
    EXTRA_PARS=$@
fi

ALPHA=${ALPHA_OPTS[${SLURM_ARRAY_TASK_ID}]}
MU_INIT=${MU_INIT_OPTS[${SLURM_ARRAY_TASK_ID}]}

HOME_DIR=$HOME
PROJ_DIR=/projects/compsci/karuturi/wangh/imaging

LOG_DIR=${PROJ_DIR}/logs/autoencoder
SRC_DIR=${PROJ_DIR}/cnn_autoencoder/src

MODEL_DIR=/projects/researchit/cervaf/models
DATA_DIR=/projects/researchit/cervaf/s3_bucket_data
CFG_DIR=/projects/researchit/cervaf/config

module load singularity

# run_command="singularity run --nv\
#  -B ${LOG_DIR}:/mnt/logs,${CFG_DIR}/:/mnt/config,${MODEL_DIR}:/mnt/model/,\
# ${DATA_DIR}:/mnt/data,${SRC_DIR}:/mnt/src/:ro\
#  /projects/researchit/cervaf/containers/cnn_autoencoder_latest.sif\
#  /mnt/src/lc_compress.py -pl -g\
#  -c /mnt/config/lc_compression_conf.json\
#  -lctg sch2_${SLURM_ARRAY_TASK_ID}\
#  -lccvs scheme_2\
#  -lcmui ${MU_INIT}\
#  -lcal ${ALPHA}\
#  -ld /mnt/logs\
#  ${EXTRA_PARS}"

# echo "$run_command"
# eval "$run_command"

# run_command="singularity run --nv\
#  -B ${LOG_DIR}:/mnt/logs,${CFG_DIR}:/mnt/config,${MODEL_DIR}:/mnt/model/,\
# ${DATA_DIR}:/mnt/data,${SRC_DIR}:/mnt/src/:ro\
#  /projects/researchit/cervaf/containers/cnn_autoencoder_latest.sif\
#  /mnt/src/lc_compress.py -pl -g\
#  -c /mnt/config/ft_compression_conf.json\
#  -lcpm /mnt/logs/cnn_autoencoder_lc_sch2_${SLURM_ARRAY_TASK_ID}.th\
#  -lctg sch2_${SLURM_ARRAY_TASK_ID}\
#  -lccvs scheme_2\
#  -lcmui ${MU_INIT}\
#  -lcal ${ALPHA}\
#  -ld /mnt/logs\
#  ${EXTRA_PARS}"

vm_root=/home/$USER/work
exampledir=$PROJ_DIR/LC-model-compression
run_command="singularity exec --nv --pwd ${vm_root} -B ${exampledir}:${vm_root} \
/projects/researchit/cervaf/containers/cnn_autoencoder_latest.sif /bin/bash examples/cvpr2020/run_debug.sh 2>&1"
echo "$run_command"
eval "$run_command"
