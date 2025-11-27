#$ -N CHiME
#$ -cwd
#$ -q long.q
#$ -l gpu=4,gpu_ram=20G
#$ -o log/$JOB_NAME_$JOB_ID.out
#$ -e log/$JOB_NAME_$JOB_ID.err

set -eux
# As Karel said don't be an idiot and use the same number of GPUs as requested
export N_GPUS=4
export $(/mnt/matylda4/kesiraju/bin/gpus $N_GPUS) || exit 1

CFG="$1"
[ -z "$1" ] && CFG="$CFG_PATH"

SRC_ROOT=/mnt/matylda5/ipoloka/projects/TS-ASR-Whisper
cd $SRC_ROOT
source configs/local_paths.sh
./python src/main.py "$CFG"
