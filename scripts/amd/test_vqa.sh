#!/bin/bash

#SBATCH -J robin_vqa
#SBATCH -o /work1/aroger/%u/job_logs/%x-%j.out
#SBATCH -e /work1/aroger/%u/job_logs/%x-%j.err
#SBATCH -t 12:00:00
#SBATCH -p mi2104x
#SBATCH -N 1


module load rocm/5.4.2
module load openmpi4/4.1.5
source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate robin

CHECKPOINT_PATH="/work1/aroger/anatolejdm/checkpoints/split_3x3_ViT_SO400M_14_SigLIP_384/finetune_split"
MODEL_BASE="/work1/aroger/anatolejdm/downloaded_models/OpenHermes-2.5-Mistral-7B"
DEVICE="cuda:0"
CONV_MODE="vicuna_v1"
TEMPERATURE=0
MAX_NEW_TOKENS=100
IMAGE_PATH="/work1/aroger/anatolejdm/data/Robin-test-data/images"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run the Python script with the parameters
python /work1/aroger/anatolejdm/robin/robin/eval/model_vqa_loader_test.py \
    --conv_mode "$CONV_MODE" \
    --temperature "$TEMPERATURE" \
    --model_base "$MODEL_BASE" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --images "$IMAGE_PATH" \
    --query_path "/work1/aroger/anatolejdm/data/Robin-test-data/blip_laion_cc_sbu_558k.json" \