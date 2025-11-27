#!/bin/bash

# Root directory of the source code.
export SRC_ROOT="/mnt/matylda5/ipoloka/projects/TS-ASR-Whisper"

# Name of the Weights & Biases project.
export WANDB_PROJECT="tsasr_whisper_update"

# Weights & Biases entity (username or team name).
export WANDB_ENTITY="butspeechfit"

# Cache directory for Hugging Face models.
export HF_HOME="/mnt/scratch/tmp/ipoloka/hf_cache"

# Set to 0 for online mode with Hugging Face Hub.
export HF_HUB_OFFLINE=0

# Add source root to the Python path.
export PYTHONPATH="$SRC_ROOT"

# Add SCTK binaries to the system path.
export PATH="/mnt/matylda5/ipoloka/utils/SCTK/bin:$PATH"

# Path for experiment outputs.
export EXPERIMENT_PATH="${SRC_ROOT}"

# Directory containing manifest files.
export MANIFEST_DIR="/mnt/scratch/tmp/ipoloka/mt_asr_data/manifests/"
export MANIFEST_DIR_DIARIZED="/mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge/diar_exp/DiariZen_base_new"


# Prefix in audio paths to be replaced.
export AUDIO_PATH_PREFIX=

# Replacement prefix for audio paths.
export AUDIO_PATH_PREFIX_REPLACEMENT=

# Path to pretrained CTC models.
export PRETRAINED_CTC_MODELS_PATH="/mnt/scratch/tmp/xkleme15/asr/chime/icassp2024_paper_models/CTC_pretrained/"

# Path to the pretrained model checkpoint.
export PRETRAINED_MODEL_PATH="/mnt/scratch/tmp/xkleme15/asr/chime/icassp2024_paper_models/large-v3_all_datasets_v1_weighted/checkpoint-6000/"

export TOKENIZERS_PARALLELISM=false

export MUSAN_ROOT="/mnt/matylda2/data/MUSAN/musan"